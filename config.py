NUM_CLASSES = 21

# ✅ FEATURE_INPUT_SIZE 별도 선언 (예: 모델 입력 feature dimension)
FEATURE_INPUT_SIZE = 21

# ✅ 실패 데이터 복사 비율 파라미터
FAIL_AUGMENT_RATIO = 3  # 기본값: 실패 데이터 3배 복사

def get_class_groups(num_classes=21, group_size=7):
    """
    ✅ 클래스 그룹화 함수 (YOPO v4.1)
    - num_classes를 group_size 크기로 나누어 그룹화
    - num_classes ≤ group_size 시 단일 그룹 반환
    - ex) num_classes=21, group_size=7 → [[0-6], [7-13], [14-20]]
    """
    if num_classes <= group_size:
        return [list(range(num_classes))]
    return [list(range(i, min(i+group_size, num_classes))) for i in range(0, num_classes, group_size)]
