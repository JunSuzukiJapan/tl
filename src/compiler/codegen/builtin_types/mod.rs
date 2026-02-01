pub mod generic;
pub mod non_generic;
pub mod resolver;

// Re-export specific modules for easier access
pub use generic::vec;
pub use generic::hashmap;
pub use generic::option;
pub use generic::result;

pub use non_generic::io;
pub use non_generic::system;
pub use non_generic::llm;
pub use non_generic::tensor;
// pub use non_generic::primitives; // Maybe? existing code didn't complain about primitives call, but let's be safe.
