# 2025-1-DLP
중간고사 범위: 15~18

CNN RNN 위주

18장 코드 보여주고 작업 해석
~ 33p 코드 














## numpy를 폐쇄망 환경에서 쓸 수 있는지 판단
### Python 실행 과정
1. **바이트코드 컴파일**: Python 소스코드(.py)는 실행 전에 바이트코드(.pyc)로 컴파일됩니다. 이 과정은 Python 인터프리터(CPython 등)에 의해 자동으로 수행됩니다. 바이트코드는 기계어가 아닌, Python Virtual Machine(PVM)이 이해할 수 있는 중간 형태입니다.
2. **PVM에서 실행**: 바이트코드는 PVM에서 한 줄씩 해석되고 실행됩니다. 이 단계는 전통적인 인터프리터 방식에 해당합니다.

### Python 코드를 EXE 파일로 빌드(예: PyInstaller 사용)
Python 스크립트와 함께 Python 인터프리터 및 필요한 라이브러리들이 패키징됩니다.
생성된 EXE 파일은 독립적으로 실행 가능하지만, 내부적으로는 여전히 Python 인터프리터를 사용하여 코드를 실행합니다. 따라서 완전한 기계어로 변환된 컴파일 언어의 EXE 파일과는 다릅니다.


> [!NOTE]    
> **동일한 기능을 파이썬으로 빌드한 EXE 파일이 컴파일 언어로 빌드한 EXE 파일보다 용량이 큰 이유**
> **1. Python 실행 환경 포함**:   
> - Python으로 만든 EXE 파일은 Python 인터프리터를 포함합니다. 이는 Python이 설치되지 않은 환경에서도 프로그램을 실행할 수 있도록 하기 위함입니다. 결과적으로, 실행 파일에 Python 런타임과 관련 파일들이 추가되어 용량이 커집니다.   
> **2. 모듈 및 라이브러리 포함**:   
> - Python 스크립트에서 사용하는 모든 모듈과 라이브러리가 EXE 파일에 포함됩니다. 심지어 사용하지 않는 모듈도 포함되는 경우가 있어, 불필요한 용량 증가를 초래할 수 있습니다.     
> **3. 컴파일 언어와의 차이**   
> 컴파일 언어(C, C++ 등)로 빌드된 EXE 파일은 소스 코드를 기계어로 변환하여 실행 환경에 의존하지 않는 독립적인 형태로 생성됩니다. 따라서 Python처럼 별도의 런타임 환경을 포함하지 않아 크기가 훨씬 작습니다.    
>    
> **Python EXE 파일의 용량을 줄이기 위한 최적화 방법**   
> 1. **PyInstaller 옵션 활용**: 불필요한 모듈 제외(--exclude-module) 또는 단일 파일 생성 옵션(--onefile) 사용.    
> 2. **가상환경 사용**: 최소한의 모듈만 포함된 가상환경에서 빌드 진행.    
> 3. **Cython 활용**: Python 코드를 C 코드로 변환 후 컴파일하여 용량을 줄일 수 있음.
> 4. **FFI 활용**:Python의 특정 라이브러리를 꼭 사용해야 할 때 컴파일 언어의 FFI(Foreign Function Interface)를 활용하여 두 언어를 병행하는 방법

### 결론
NumPy는 내부적으로 BLAS와 LAPACK 같은 고성능 C 라이브러리를 사용하며 내부적으로 컴파일된 코드(C/C++ 기반)와 Python 바인딩을 통해 실행됩니다. API 호출 등을 통한 외부 리소스를 활용하지 않습니다.
따라서 폐쇄망 환경에서도 사용 가능합니다.
