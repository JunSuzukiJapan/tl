struct Item {
    name: String,
    value: i64
}

struct Container {
    items: Vec<Item>
}

fn get_item_free(c: Container, idx: i64) -> String {
    let items = c.items;
    let item = items.get(idx).unwrap();
    item.name
}

fn main() {
    let mut items: Vec<Item> = Vec::new();
    let item = Item { name: "hello", value: 42 };
    items.push(item);
    let c = Container { items: items };
    let name1 = get_item_free(c, 0);
    println("Free function: {}", name1);
}
