from collections import deque
from typing import Optional, Iterable

features = [
    {
        "trigram_non_zero_0x00000c": 0.00013152455751381007,
        "trigram_non_zero_0x00000d": 0.00016910300251775582,
    },
    {"trigram_non_zero_0x00000c": 0.00016880705353780628, "trigram_non_zero_0x00000e": 5.194063185778655e-05},
    {"trigram_non_zero_0x00000d": 0.00016880705353780628, "trigram_non_zero_0x00000e": 5.194063185778655e-05},
]


class Node:
    def __init__(self, data: object, prev: Optional["Node"], next: Optional["Node"]) -> None:
        self.data = data
        self.prev = prev
        self.next = next


class KeyedLinkedList:
    def __init__(self, data: Iterable[str]) -> None:
        key_node_dict, first_node = self.__key_node_dict_from_iterable(data)

        self.__key_node_dict = key_node_dict
        self.first_node = first_node

    @staticmethod
    def __key_node_dict_from_iterable(data: Iterable[str]) -> tuple[dict[str, Node], Node]:
        key_node_dict = dict()
        prev_node = None
        first_node = None
        for node_data in data:
            node = Node(node_data, None, None)
            key_node_dict[node_data] = node
            if first_node is None:
                first_node = node
            if prev_node is not None:
                node.prev = prev_node
                prev_node.next = node
            prev_node = node
        return key_node_dict, first_node

    def get_node(self, key: str) -> Optional[Node]:
        return self.__key_node_dict.get(key)

    def insert_before_node(self, data: str, node: Node) -> None:
        prev_node = node.prev
        new_node = Node(data, prev_node, node)
        node.prev = new_node
        self.__key_node_dict[data] = new_node

        if prev_node is None:
            self.first_node = new_node
        else:
            prev_node.next = new_node

    def insert_after_node(self, data: str, node: Node) -> None:
        next_node = node.next
        new_node = Node(data, node, next_node)
        node.next = new_node
        self.__key_node_dict[data] = new_node

        if next_node is not None:
            next_node.prev = new_node

    def insert_before_key(self, data: str, key: str) -> None:
        node = self.get_node(key)
        if node is None:
            raise KeyError(f"node with key does not exist: {key}")
        self.insert_before_node(data, node)

    def insert_after_key(self, data: str, key: str) -> None:
        node = self.get_node(key)
        if node is None:
            raise KeyError(f"node with key does not exist: {key}")
        self.insert_after_node(data, node)

    def __iter__(self):
        return _KeyedLinkedListIter(self)

    def __contains__(self, key: str) -> bool:
        return key in self.__key_node_dict

    def __getitem__(self, key: str) -> Node:
        return self.__key_node_dict[key]


class _KeyedLinkedListIter:
    def __init__(self, keyed_linked_list: KeyedLinkedList) -> None:
        self.keyed_linked_list = keyed_linked_list
        self.cur_iter_node = keyed_linked_list.first_node

    def __next__(self):
        if self.cur_iter_node is None:
            raise StopIteration
        cur_node = self.cur_iter_node
        self.cur_iter_node = self.cur_iter_node.next
        return cur_node


def labels_sorted(features: list[dict[str, float]], included_labels: set[str]) -> list[str]:
    features_included_labels = [
        list(filter(lambda key: key in included_labels, feature.keys())) for feature in features
    ]

    features_included_labels_iter = iter(features_included_labels)
    keyed_linked_list = KeyedLinkedList(next(features_included_labels_iter))

    # Labels not found. No info to determine the position of these labels. They must be inserted any place.
    labels_not_found = set()
    for label_list in features_included_labels_iter:
        prev_label = None
        for label in label_list:
            cur_prev_label = prev_label
            prev_label = label
            if label in keyed_linked_list:
                continue


if __name__ == "__main__":
    labels_sorted(
        features, {"trigram_non_zero_0x00000c", "trigram_non_zero_0x00000d", "trigram_non_zero_0x00000e"}
    )
