use core::{fmt, hash::{Hash, Hasher}};


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct AtomId(pub usize);

impl fmt::Display for AtomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "atom#{:x}", self.0)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Atom {
    Nil,
    True,
    False,
    Custom(AtomId),
}

impl Hash for Atom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Atom::Nil => "Nil".hash(state),
            Atom::True => "True".hash(state),
            Atom::False => "False".hash(state),
            Atom::Custom(id) => id.hash(state),
        }
    }
}

impl Atom {
    pub fn is_falsey(&self) -> bool {
        matches!(self, Atom::False | Atom::Nil)
    }

    pub fn is_false(&self) -> bool {
        matches!(self, Atom::False)
    }

    pub fn is_true(&self) -> bool {
        matches!(self, Atom::True)
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Atom::Nil)
    }
}

impl fmt::Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Nil => write!(f, "Nil"),
            Atom::True => write!(f, "True"),
            Atom::False => write!(f, "False"),
            Atom::Custom(id) => write!(f, "{:?}", id),
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Nil => write!(f, "Nil"),
            Atom::True => write!(f, "True"),
            Atom::False => write!(f, "False"),
            Atom::Custom(s) => write!(f, "{}", s),
        }
    }
}
