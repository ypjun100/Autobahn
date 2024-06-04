`Vatsal Chheda`

# 요약
* 수치형과 범주형 변수의 결측값에 대한 해결방법 제시
* 변수의 불균형 문제를 다루기 위해 SMOTE 기법과 유사한 ADASYN 사용
---

# Introduction
* 본 논문에서 제안하는 파이프라인은 데이터셋을 가져와 이름이 없거나 날짜 혹은 ID 형식의 열을 제거함
* 가격을 나타낼 때 사용되는 $와 같은 통화 기호도 제거됨
* 수치형 변수에 대해 데이터 대치 및 스케일링 작업이 완료되면, 범주형 변수와 타깃 변수에 대한 인코딩 진행
* 데이터가 불균형으로 이루어져 있다면 ADASYN을 사용하여 샘플링 후 80-20의 셋으로 분할

# Methodology
#### Data & Data Preprocessing
* **Simple Imputer** : 본 논문에서는 결측치를 처리하기 위해 Scikit-learn의 Simple Imputer를 이용하였고, 사용된 전략은 'mean'
* **One Hot Encoding** : 범주형 데이터의 각 범주에 대한 가중치를 부여하지 않고도 모델이 이해할 수 있는 형식
* **Label Encoding** : 원 핫 인코딩과 달리 각 범주에 해당하는 숫자로 라벨을 붙여 모델이 이해할 수 있게끔 변환
* **Standard Scaler** : 분포의 평균이 0이고, 표준 편차가 1이 되도록 변환
* **ADASYN** : 불균형한 데이터 분포를 가지고 있는 데이터셋에 적용하는 방법론
#### Implementation and Results
![[Pasted image 20240603221644.png]]
* 여러 분류 데이터셋을 무작위로 가져와 제안된 파이프라인에서 테스트 진행
* 데이터셋은 웹 애플리케이션에 업로드되고, 타깃 변수의 이름이 입력됨
* 우선, 열 이름이 정의되지 않았거나 날짜가 포함된 열은 모두 삭제됨
* 통화 기호가 포함된 열이 있으면 해당 기호를 제거하고, Float로 변환됨
* 이후 데이터셋은 수치형 변수 열과 범주형 범수 열과 마지막으로 종속 변수 열로 나뉘게 됨
* 먼저, 수치형 변수의 결측치는 Simple Imputer의 평균 전략을 사용해서 값이 대체됨
* 범주형 변수의 경우 결측값이 있는 행은 삭제되고, 원-핫 인코딩을 진행 (필요한 경우 라벨 인코딩 진행)
* 그리고 데이터셋이 균형을 이루고 있는지 확인하며, 데이터를 최소로 가진 클래스와 최대로 가진 클래스의 비율이 0.3 미만인 경우 ADASYN을 사용하여 샘플링 진행
* 마지막으로 독립 변수의 스케일링은 표준 스케일러를 사용