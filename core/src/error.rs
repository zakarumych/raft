use core::fmt;

use crate::string::RcStr;


#[derive(Clone, Debug)]
pub enum RuntimeError {
    UnboundIdentifier(RcStr),
    NotAFunction(RcStr),
    TypeError(RcStr),
    IndexError(RcStr),
    FieldError(RcStr),
    Other(RcStr),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::UnboundIdentifier(s) => write!(f, "Unbound identifier: {}", s),
            RuntimeError::NotAFunction(s) => write!(f, "Not a function: {}", s),
            RuntimeError::TypeError(s) => write!(f, "Type error: {}", s),
            RuntimeError::IndexError(s) => write!(f, "Index error: {}", s),
            RuntimeError::FieldError(s) => write!(f, "Field error: {}", s),
            RuntimeError::Other(s) => write!(f, "{}", s),
        }
    }
}
