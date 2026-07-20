use crate::{
    coro::{CoroKind, CoroStatus, RcCoro},
    error::RuntimeError,
    host::Host,
    list::RcList,
    record::RcRecord,
    string::RcStr,
    val::{Val, ValEnum},
};

/// The `for` statement's iteration protocol, shared by every execution
/// mode: lists yield their elements, records yield one single-field record
/// per entry, and generator objects are resumed once per step (needing the
/// live host, which is why this isn't a plain `Iterator` - see
/// [`ValsIter::next`]).
pub enum ValsIter {
    List(RcList, usize),
    Record(RcRecord, usize),
    Coro(RcCoro),
}

/// What one [`ValsIter::step`] produced.
pub enum ValsIterStep {
    /// The next item.
    Item(Val),
    /// Clean exhaustion - or a failed coroutine step, with the host's
    /// pending-error state set (callers check).
    End,
    /// An async-gen coroutine must wait before it can produce the next
    /// item - suspend the (async) consumer and re-`step` later.
    Pending,
}

impl ValsIter {
    pub fn new(v: &Val) -> Result<ValsIter, RuntimeError> {
        match v.unpack() {
            ValEnum::List(l) => Ok(ValsIter::List(l, 0)),
            ValEnum::Record(r) => Ok(ValsIter::Record(r, 0)),
            ValEnum::Coro(c) => match c.kind() {
                Some(CoroKind::Gen) | Some(CoroKind::AsyncGen) => Ok(ValsIter::Coro(c)),
                _ => Err(RuntimeError::TypeError(
                    "cannot iterate an async coroutine".into(),
                )),
            },
            _ => Err(RuntimeError::TypeError(
                "iteration on non-heap value".into(),
            )),
        }
    }

    /// One iteration step. `Pending` only ever comes from an async-gen
    /// coroutine that must wait - consumers in an async context suspend on
    /// it and re-`step` later; everywhere else it's an error (see
    /// [`ValsIter::next`]). An `End` from a *failed* coroutine step leaves
    /// the host's pending-error state set - callers check it before
    /// treating `End` as a clean end.
    pub fn step(&mut self, host: &mut Host) -> Result<ValsIterStep, RuntimeError> {
        match self {
            ValsIter::List(l, pos) => {
                let Some(item) = l.get(*pos) else {
                    return Ok(ValsIterStep::End);
                };
                *pos += 1;
                Ok(ValsIterStep::Item(item))
            }
            ValsIter::Record(r, pos) => {
                let Some((key, value)) = r.entry_at(*pos) else {
                    return Ok(ValsIterStep::End);
                };
                *pos += 1;
                Ok(ValsIterStep::Item(Val::record([(RcStr::new(key), value)])))
            }
            ValsIter::Coro(c) => match c.resume(host, 0) {
                CoroStatus::Yielded => Ok(ValsIterStep::Item(host.stack().pop())),
                CoroStatus::Done => Ok(ValsIterStep::End),
                CoroStatus::Pending => match c.kind() {
                    Some(CoroKind::AsyncGen) => Ok(ValsIterStep::Pending),
                    _ => Err(RuntimeError::TypeError(
                        "generator coroutine reported pending".into(),
                    )),
                },
            },
        }
    }

    /// The next item, or `Ok(None)` when exhausted - [`ValsIter::step`]
    /// for synchronous consumers, where `Pending` (an async-gen coroutine
    /// that must wait) has nowhere to suspend to and is an error. A
    /// coroutine step that *fails* also comes back `Ok(None)` (resume
    /// reported `Done` without pushing) with the host's pending-error
    /// state set - callers must check that state after every `next` before
    /// treating `None` as a clean end.
    pub fn next(&mut self, host: &mut Host) -> Result<Option<Val>, RuntimeError> {
        match self.step(host)? {
            ValsIterStep::Item(v) => Ok(Some(v)),
            ValsIterStep::End => Ok(None),
            ValsIterStep::Pending => Err(RuntimeError::TypeError(
                "cannot iterate an async generator outside an async context".into(),
            )),
        }
    }

    pub fn iter<'a, 'b>(&'a mut self, host: &'a mut Host<'b>) -> IterVals<'a, 'b> {
        IterVals { host, iter: self }
    }
}

pub struct IterVals<'a, 'b> {
    host: &'a mut Host<'b>,
    iter: &'a mut ValsIter,
}

impl Iterator for IterVals<'_, '_> {
    type Item = Result<Val, RuntimeError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next(self.host).transpose()
    }
}
