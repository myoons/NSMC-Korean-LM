## _Hugging Face KoELECTRA [1] 를 사용하여 NSMC [2] 문제를 풀어보았습니다._

<br/>

## Author 👤 : **Yoonseo Kim**

[_Github_](https://github.com/myoons)  
[_Website_](https://ainote.tistory.com/)
<br/>
<br/>

## Dataset

_NSMC는 한국어 영화 리뷰 데이터셋으로 Naver Movie 댓글을 수집한 데이터셋입니다._  
_평점 1-4 는 부정(0) 으로, 9-10은 긍정(1) 으로 라벨링 되어 Train 15만 문장, Test 5만 문장으로 구성되어있습니다._  
_저는 NaN (비어있는 리뷰) 와 중복을 제거하여 Train 146,182 / Test 49,157 문장을 사용하였습니다._  
<br/>

_**예시**_
```bazaar
6483659 사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다  1  
5403919 막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.     0
```

<br/>
<br/>


## **Setup**

### _**1. 레포지토리 다운로드**_

_`git clone` 을 사용하여 레포지토리를 다운로드 받는다._

    git clone https://github.com/myoons/NSMC-Korean-LM.git


<br/>

### _**2. Dataset 설정하기**_
_[NSMC Dataset](https://github.com/e9t/nsmc) 로부터 ratings_train.txt , rating_test.txt 를 다운받는다._  
_아래 디렉터리 구조와 동일하게 txt 파일을 위치시킨다._

    ├── data
        ├── ratings_train.txt 
        ├── ratings_test.txt
        └── eval.txt

<br/>

### _**3. 학습하기**_
_아래 Shell 파일을 실행하면 KoELECTRA 가 학습되며, logs 폴더에 학습된 모델이 에폭 단위로 저장됩니다._
_GPU 가 없다면 --cuda 옵션을 제거하세요, 하지만 GPU 로 학습하시는 것을 추천드립니다!_  

    bash run_KoELECTRA.sh

<br/>
<br/>

## **Tensorboard**
_Tensorboard 를 사용하시면 학습 / 평가 추이를 확인하실 수 있습니_  
_아래는 Tensorboard 를 사용하여 확인한 그래프입니다._

    tensorboard --logdir ./logs

<br/>

![train_test_graph](https://user-images.githubusercontent.com/67945103/106461189-c8877980-64d7-11eb-884d-944d9011cc47.png)


<br/>
<br/>

## **평가하기**
_rating_test.txt 외에 data/ 에 있는 eval.txt 를 사용하여 학습된 모델을 평가할 수 있습니다._  
_eval.txt 는 최근 네이버 영화 리뷰에서 50개 (긍정 25개, 부정 25개) Sampling 해서 만든 데이터셋입니다._
_구성은 NSMC Dataset 과 동일합니다._  
_NSMC_eval.txt 는 eval.txt 에 대한 평가 결과입니다._  
<br/>

_**예시**_
```bazaar
[1] 홍콩을 버린자의 영화.관객이 그를 버렸다.
 - Label : -
 - Prediction : -
 - Correct : O
 - softmax(NN(x))[label_c] :0.9645477533340454
```

<br/>
<br/>

## Reference
[0] _**[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2001.07685)**_

[1] _**[KoELECTRA - Github](https://github.com/monologg/KoELECTRA)**_

[2] _**[NSMC Dataset](https://github.com/e9t/nsmc)**_