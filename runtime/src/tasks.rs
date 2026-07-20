//! The single-threaded task executor and its FFI-safe waker.
//!
//! Spawning an async value creates a task: the pollable future plus an
//! owned [`RtWaker`] - a thin-pointer FFI waker (vtable in the
//! allocation's prefix, see `raft-ffi`'s async section) holding the task's
//! id and a weak handle to the executor's ready queue. While a task is
//! polled its waker is ambient on the host (`RawHost::waker`); a leaf
//! async value that comes back pending clones it out
//! ([`raft_core::rc::Host::waker`]) and wakes it when its result is ready,
//! which pushes the task id back onto the ready queue. [`Runtime::block_on`]
//! drains that queue until the target task resolves.
//!
//! Everything here is single-threaded: wakers must only be cloned, woken
//! and dropped on the runtime's own thread (despite `core::task::Waker`'s
//! nominal `Send + Sync` once adapted).

use alloc::{
    boxed::Box,
    sync::{Arc, Weak},
};
use core::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll,},
};

use futures_util::task::AtomicWaker;

use raft_core::{
    CoroKind, CoroStatus, Coroutine, RcCoro, RuntimeError, Val, ValEnum, ffi, FfiWaker,
};

use crate::Runtime;

/// Identifies one spawned task for the lifetime of its runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TaskId(pub u64);

struct WakeQueue {
    queue: amity::flip_queue::FlipQueue<TaskId>,
    waker: AtomicWaker,
}

/// Task ids whose futures are ready to be polled again. Shared: the
/// executor owns it strongly, every task waker weakly - a wake after the
/// runtime is gone is a silent no-op.
pub struct ReadyQueue {
    wake: Arc<WakeQueue>,
    owned: amity::ring_buffer::RingBuffer<TaskId>,
}

impl ReadyQueue {
    pub fn new() -> Self {
        ReadyQueue {
            wake: Arc::new(WakeQueue {
                queue: amity::flip_queue::FlipQueue::new(),
                waker: AtomicWaker::new(),
            }),
            owned: amity::ring_buffer::RingBuffer::new(),
        }
    }

    fn pop(&mut self) -> Option<TaskId> {
        if self.owned.is_empty() {
            self.wake.queue.swap_buffer(&mut self.owned);
        }

        self.owned.pop()
    }
}

pub(crate) struct TaskEntry {
    future: RcCoro,
    /// The task's waker - one strong reference owned by the entry
    /// (dropped with it); every clone handed out to leaf futures is its
    /// own reference.
    waker: FfiWaker,
}

// ---------------------------------------------------------------------
// RtWaker: the executor's concrete FFI waker. `#[repr(C)]` with the
// `WakerHeader` first, so a pointer to the struct *is* the thin
// `WakerPtr` the FFI passes around.
// ---------------------------------------------------------------------

#[repr(C)]
struct RtWaker {
    header: ffi::WakerHeader,
    task: TaskId,
    queue: Weak<WakeQueue>,
}

static RT_WAKER_VTABLE: ffi::WakerVTable = ffi::WakerVTable {
    wake: rt_waker_wake,
    destroy: rt_waker_destroy,
};

fn new_rt_waker(task: TaskId, queue: &ReadyQueue) -> FfiWaker {
    let raw = Box::into_raw(Box::new(RtWaker {
        header: ffi::WakerHeader {
            vtable: &RT_WAKER_VTABLE,
            strong: AtomicUsize::new(1),
        },
        task,
        queue: Arc::downgrade(&queue.wake),
    }));
    // SAFETY: `raw` is a fresh, live allocation whose header is its first
    // field (`#[repr(C)]`), carrying one strong reference.
    unsafe { FfiWaker::from_raw(ffi::WakerPtr::new_unchecked(raw as *mut ffi::WakerHeader)) }
}

unsafe extern "C" fn rt_waker_wake(ptr: ffi::WakerPtr) {
    let w = unsafe { &*(ptr.as_ptr() as *const RtWaker) };
    if let Some(queue) = w.queue.upgrade() {
        queue.queue.push_sync(w.task);
        queue.waker.wake();
    }
}

unsafe extern "C" fn rt_waker_destroy(ptr: ffi::WakerPtr) {
    {
        let w = unsafe { &*(ptr.as_ptr() as *const RtWaker) };
        debug_assert_eq!(w.header.strong.load(Ordering::Relaxed), 0,);
    }
    drop(unsafe { Box::from_raw(ptr.as_ptr() as *mut RtWaker) });
}

// ---------------------------------------------------------------------
// The executor: spawn / poll / block_on on Runtime.
// ---------------------------------------------------------------------

impl Runtime {
    /// Create a task driving `future` (a `Val::Async`, e.g. what calling
    /// an `async fn` returns) and queue it for its first poll. The task
    /// runs when [`Runtime::block_on`] drains the ready queue.
    pub fn spawn(&mut self, future: Val) -> Result<TaskId, RuntimeError> {
        let f = match future.unpack() {
            ValEnum::Coro(c) if c.kind() == Some(CoroKind::Async) => c,
            _ => {
                return Err(RuntimeError::TypeError(
                    "spawn expects an async value".into(),
                ));
            }
        };
        let id = TaskId(self.next_task);
        self.next_task += 1;
        let waker = new_rt_waker(id, &self.ready);
        self.tasks.insert(id, TaskEntry { future: f, waker });
        self.ready.owned.push(id);
        Ok(id)
    }

    /// Poll one task, its waker ambient on the host for the duration.
    /// `Ok(Some(v))` = resolved (task removed); `Ok(None)` = pending, or a
    /// stale wake for an already-finished task.
    fn poll_task(&mut self, id: TaskId) -> Result<Poll<Val>, RuntimeError> {
        let Some(entry) = self.tasks.get(&id) else {
            return Ok(Poll::Pending); // stale wake - the task already finished
        };
        let future = entry.future.clone();
        let waker_ptr = FfiWaker::as_raw(&entry.waker);

        let prev = self.host.waker;
        self.host.waker = waker_ptr.as_ptr();
        let out = crate::await_step(self, &future);
        self.host.waker = prev;

        match out {
            Ok(Poll::Pending) => Ok(Poll::Pending),
            Ok(Poll::Ready(v)) => {
                self.tasks.remove(&id);
                Ok(Poll::Ready(v))
            }
            Err(e) => {
                self.tasks.remove(&id);
                Err(e)
            }
        }
    }

    /// Continuously poll tasks until the one spawned for `future` resolves,
    /// returning its value or error.
    pub async fn eval_async(&mut self, future: Val) -> Result<Val, RuntimeError> {
        struct EvalAsync<'a> {
            rt: &'a mut Runtime,
            target: TaskId,
        }

        impl core::future::Future for EvalAsync<'_> {
            type Output = Result<Val, RuntimeError>;

            fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<Val, RuntimeError>> {
                let target = self.target;
                let rt = &mut *self.rt;

                while let Some(id) = rt.ready.pop() {
                    match rt.poll_task(id) {
                        Ok(Poll::Ready(v)) => {
                            rt.tasks.remove(&target);
                            if id == target {
                                return Poll::Ready(Ok(v));
                            }
                        }
                        Ok(Poll::Pending) => {}
                        Err(e) => {
                            rt.tasks.remove(&id);
                            return Poll::Ready(Err(e));
                        }
                    }
                }

                rt.ready.wake.waker.register(cx.waker());
                Poll::Pending
            }
        }

        let target = self.spawn(future)?;

        EvalAsync { rt: self, target }.await
    }
}

// ---------------------------------------------------------------------
// FutureAsync: adapt an ordinary Rust future into the object model's
// async-kind `Coroutine` - it polls with a real `core::task::Waker` built
// from the ambient host waker, so wakes it schedules reach the executor.
// ---------------------------------------------------------------------

struct FutureAsync<F> {
    fut: RefCell<Option<Pin<Box<F>>>>,
}

impl<F> Coroutine for FutureAsync<F>
where
    F: Future<Output = Result<Val, RuntimeError>> + 'static,
{
    fn resume(&self, host: &mut raft_core::Host, args: usize) -> CoroStatus {
        debug_assert_eq!(args, 0, "async coroutines take no resume arguments");
        let Some(mut fut) = self.fut.borrow_mut().take() else {
            // resumed after the resolution was delivered (or after a failure):
            // the final protocol step
            return CoroStatus::Done;
        };
        let waker = host.rust_waker();
        let mut cx = Context::from_waker(&waker);
        match fut.as_mut().poll(&mut cx) {
            Poll::Pending => {
                *self.fut.borrow_mut() = Some(fut);
                CoroStatus::Pending
            }
            // the future is consumed either way, so the next resume
            // reports the final Done
            Poll::Ready(Ok(v)) => {
                host.stack().push(v);
                CoroStatus::Yielded
            }
            Poll::Ready(Err(e)) => {
                // SAFETY: as `AstFn::call`'s.
                let rt: &mut Runtime = unsafe { &mut *(host.as_raw() as *mut Runtime) };
                rt.set_error(e);
                CoroStatus::Done
            }
        }
    }

    fn debug_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "<async (host)>")
    }
}

/// Wrap an ordinary Rust future as an async-kind `Val::Coro` - the
/// host-side way to hand asynchronous work to Raft code
/// (`x = await host_thing`).
pub fn async_val(fut: impl Future<Output = Result<Val, RuntimeError>> + 'static) -> Val {
    Val::from(ValEnum::Coro(RcCoro::new(
        CoroKind::Async,
        FutureAsync {
            fut: RefCell::new(Some(Box::pin(fut))),
        },
    )))
}
