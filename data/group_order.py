# predict_trigger가 기대하는 인터페이스를 utils의 싱글톤에 래핑
from typing import List
from data.utils import GROUP_MGR

class GroupOrderManager:
    def __init__(self):
        self._mgr = GROUP_MGR
        # 유효성: 8그룹×5심볼 고정 확인
        groups = getattr(self._mgr, "groups", []) or []
        if len(groups) != 8 or any(len(g) != 5 for g in groups):
            raise RuntimeError(f"[group_order] 그룹 불일치: {len(groups)} groups / sizes={[len(g) for g in groups]}")

    # predict_trigger가 호출하는 메서드들
    def current_group_index(self) -> int:
        return self._mgr.current_index()

    def get_group_symbols(self, idx: int) -> List[str]:
        return self._mgr.groups[idx] if 0 <= idx < len(self._mgr.groups) else []

    def get_current_group_symbols(self) -> List[str]:
        return self._mgr.current_group()

    # 선택적 호환
    def mark_symbol_trained(self, symbol: str) -> None:
        self._mgr.mark_symbol_trained(symbol)

    def ready_for_group_predict(self) -> bool:
        return self._mgr.ready_for_group_predict()

    def mark_group_predicted(self) -> None:
        self._mgr.mark_group_predicted()
