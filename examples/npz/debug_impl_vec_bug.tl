// Minimal reproduction of impl Vec<Struct>.get().unwrap() type inference bug

struct Item {
    name: String,
    value: i64
}

struct Container {
    items: Vec<Item>
}

// This works (free function)
fn get_item_free(c: Container, idx: i64) -> String {
    let items = c.items;
    let item = items.get(idx).unwrap();
    item.name
}

impl Container {
    // This fails with "Field access on non-struct type Void"
    fn get_item(self, idx: i64) -> String {
        let items = self.items;
        let item = items.get(idx).unwrap();
        item.name
    }
}

fn main() {
    let mut items: Vec<Item> = Vec::new();
    let item = Item { name: "hello", value: 42 };
    items.push(item);
    let c = Container { items: items };

    // Free function works
    let name1 = get_item_free(c, 0);
    println("Free function: {}", name1);

    // Method should also work
    let name2 = c.get_item(0);
    println("Method: {}", name2);
}
