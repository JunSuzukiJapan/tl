// Users
user(alice).
user(bob).
user(carol).
user(dave).

// Roles
role(alice, admin).
role(bob, editor).
role(carol, viewer).
role(dave, intern).

// Role inheritance (role -> parent role)
inherits(admin, editor).
inherits(editor, viewer).

// Permissions
perm(admin, delete_users).
perm(admin, deploy).
perm(editor, write_docs).
perm(editor, merge_pr).
perm(viewer, read_docs).

// Optional direct grants
grant(dave, read_docs).
