## YOLOv2 and Keras
#### 객체 예측의 속도 개선

### 개요
1. 첫 프레임에서 n개의 박스 읽어들이고 n개의 박스를 배열 형태로 real_keras_input 리스트에 저장
    1-1) real_keras_input 리스트의 박스들을 모델에 집어넣고 예측값을 받는다
    1-2)

2. 다음 프레임에서 m개의 박스를 읽어들인다
    2-1) n = m일 때
        2-1-1) n개의 각 박스들의 위치가 다 비슷할 때
        2-1-2) n개 중 r개의 박스의 위치가 다를 때
    2-2) n != m일 때
        캐시 사용 !

### 주의사항
* 예측을 잘못하면 그다음 수십~수백 프레임의 라벨링이 전부 틀려버림

### 캐시
1. 동작
    첫 프레임
        모두 캐시에 입력
    두 번째 프레임부터
        모두 동일한 객체일 때
            cache_input 함수를 통해 캐시에 입력
        객체가 다를 때
            is_in_cache 함수를 통해 캐시 히트 판별
                히트일 때
                    블록 히트를 올리고 라벨링
                    단, 캐시 삭제가 무조건 FIFO 이기 때문에 히트 횟수 / 히트 레이트를 기록하는 것은 캐시 효과를 가늠하는 용도일 뿐이다. 오히려 캐시 히트를 판별하는데 사용되는 불필요한 루프는 전체 프로그램의 성능 저하를 가져올 수 있기 때문에 실제 환경에서는 삭제하는 것이 바람직하다.
                미스일 때
                    캐시가 가득 찼을 때
                        cache_delete 함수를 통해 캐시 비움
                    캐시에 여유가 있을 때
                        새 블록 추가

2. 함수 상세
* cache_input (obj)
* cache_delete ( )
    캐시 리스트에서 가장 오래된 블록부터 삭제 (FIFO)
    프로그램 특성상 최신 블록이 참조될 확률이 무조건 높기 때문에 캐시의 히트 횟수를 고려할 필요 없음
* cache_manage (objects)

3. 로그
* 캐시 도입 전
    - 모델을 이용한 예측에는 객체 개수와 크게 관계 없이 약 10초 가량이 소요

* 캐시 도입 후
    - predict 없는 프레임은 약 0.002초 가량 소요
    - prediction에 10초~30초 가량 소요
    - 전체 625프레임 중 prediction 65회
    - 캐시 traverse : 역순으로 들여다보면 더 빠르게 찾을 가능성이 높음

* 기타
    - 로그 찍는 방식