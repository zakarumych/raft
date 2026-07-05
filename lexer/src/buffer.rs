use alloc::rc::Rc;

// Cheaply cloneable buffer for parsing.
#[derive(Clone)]
pub(crate) struct Buffer {
    string: Rc<str>,
}

impl Buffer {
    pub fn from_str(s: &str) -> Self {
        Buffer {
            string: Rc::from(s),
        }
    }

    pub fn as_str(&self) -> &str {
        self.string.as_ref()
    }
}
