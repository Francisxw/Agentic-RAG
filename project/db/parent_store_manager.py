import re
import json
import config
from utils import clear_directory_contents
from pathlib import Path
from typing import List, Dict


class ParentStoreManager:
    """父块存储管理器

    负责在文件系统中持久化存储和检索 RAG 系统的父块（Parent Chunks）。
    每个父块以 JSON 文件形式保存，包含页面内容和元数据。
    采用层级索引策略：子块用于精确检索，父块用于提供丰富上下文。
    """

    __store_path: Path  # 父块存储目录路径

    def __init__(self, store_path=config.PARENT_STORE_PATH):
        """初始化父块存储管理器

        Args:
            store_path: 父块 JSON 文件的存储目录路径，默认使用配置文件中的 PARENT_STORE_PATH
        """
        self.__store_path = Path(store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)

    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        """保存单个父块到文件系统

        将父块的页面内容和元数据序列化为 JSON 文件，以 parent_id 命名。

        Args:
            parent_id: 父块唯一标识符（如 "document_parent_0"）
            content: 父块的页面文本内容
            metadata: 父块的元数据字典（包含 source 文件名、parent_id 等）
        """
        file_path = self.__store_path / f"{parent_id}.json"
        file_path.write_text(
            json.dumps({"page_content": content, "metadata": metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def save_many(self, parents: List) -> None:
        """批量保存多个父块

        Args:
            parents: 元组列表，每个元组为 (parent_id, document)
                     其中 document 为 LangChain Document 对象，
                     包含 page_content 和 metadata 属性
        """
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)

    def load(self, parent_id: str) -> Dict:
        """加载单个父块的原始 JSON 数据

        支持传入带 .json 后缀或不带后缀的 parent_id。

        Args:
            parent_id: 父块唯一标识符

        Returns:
            包含 page_content 和 metadata 两个字段的字典
        """
        file_path = self.__store_path / (
            parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        )
        return json.loads(file_path.read_text(encoding="utf-8"))

    def load_content(self, parent_id: str) -> Dict:
        """加载单个父块并统一返回格式

        将 load() 返回的原始数据转换为统一格式，
        将 page_content 重命名为 content 字段。

        Args:
            parent_id: 父块唯一标识符

        Returns:
            包含 content、parent_id、metadata 三个字段的字典
        """
        data = self.load(parent_id)
        return {
            "content": data["page_content"],
            "parent_id": parent_id,
            "metadata": data["metadata"]
        }

    @staticmethod
    def _get_sort_key(id_str):
        """从 parent_id 中提取排序序号

        从形如 "document_parent_0" 的 ID 中提取末尾的数字部分，
        用于按创建顺序对父块进行排序。若无法匹配则返回 0。

        Args:
            id_str: 父块 ID 字符串

        Returns:
            提取出的整数序号，匹配失败时返回 0
        """
        match = re.search(r'_parent_(\d+)$', id_str)
        return int(match.group(1)) if match else 0

    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        """批量加载多个父块并按 parent_id 中的序号排序

        自动去重，然后按 parent_id 末尾的数字升序排列。

        Args:
            parent_ids: 父块 ID 列表

        Returns:
            按序号排序后的父块内容字典列表
        """
        unique_ids = set(parent_ids)
        return [self.load_content(pid) for pid in sorted(unique_ids, key=self._get_sort_key)]

    def clear_store(self) -> None:
        """清空所有已保存的父块文件

        保留存储目录本身，仅删除其中的所有 JSON 文件。
        """
        self.__store_path.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.__store_path)
