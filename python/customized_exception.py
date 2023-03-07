class NoFaceException(Exception):
    def __init__(self):
        super().__init__('얼굴이 존재하지 않습니다.')

class LowEmotionError(Exception):
    def __init__(self):
        super().__init__('감정 confidence가 0.4이하입니다.')