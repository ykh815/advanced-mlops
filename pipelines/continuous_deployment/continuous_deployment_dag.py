from datetime import datetime

import bentoml
import pendulum
import requests
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)
from airflow.sdk import DAG, Variable, get_current_context

from utils.callbacks import failure_callback, success_callback

local_timezone = pendulum.timezone("Asia/Seoul")
airflow_dags_path = Variable.get("AIRFLOW_DAGS_PATH")


def get_branch_by_api_status() -> list[str] | str:
    """
    API 상태를 확인하여 분기를 결정하는 함수.
    Airflow 3.x에서는 provide_context가 제거되어 함수 시그니처 변경 불필요.
    """
    try:
        response = requests.get("http://localhost:3000/healthz")
        if response.status_code == 200:
            # TODO: 헬스체크의 응답이 올바르게 왔다면 다음 Task를 실행해야 함
            # "get_deployed_model_creation_time", "get_latest_trained_model_creation_time" 를 실행해야 함
            return
        else:
            return "deploy_new_model"
    except Exception as e:
        print(f"API 통신이 이루어지지 않았습니다.: {e}")
        return "deploy_new_model"


def get_deployed_model_creation_time() -> datetime | None:
    """이미 배포된 모델의 `creation_time`을 조회합니다."""
    try:
        response = requests.post("http://localhost:3000/metadata")
        if response.status_code == 200:
            # TODO: 메타데이터 조회 응답이 올바르게 왔다면 메타데이터 내 모델의 생성 시간(creation_time)을 datetime 객체로 반환해야 함
            return
        else:
            print(
                f"`creation_time`을 불러올 수 없습니다.: {response.status_code}"
            )
            return None
    except Exception as e:
        print(f"배포된 모델의 API를 받아오지 못했습니다.: {e}")
        return None


def get_latest_trained_model_creation_time() -> datetime | None:
    """로컬 저장소에 저장된 최신 학습 모델의 `creation_time` 조회합니다."""
    try:
        bento_model = bentoml.models.get("credit_score_classification:latest")
        # TODO: bento_model의 creation_time의 timezone 정보를 제거하고 반환
        return
    except Exception as e:
        print(f"Error getting latest trained model creation time: {e}")
        return None


def decide_model_update():
    """
    현재 배포된 모델과 로컬 최신 학습 모델의 creation_time 비교.
    배포된 모델이 오래되었으면 새로운 모델을 배포하도록 결정.

    Airflow 3.x에서는 명시적으로 ti를 인자로 받는 것을 권장하지만,
    기존 코드와의 호환성을 위해 이 방식도 계속 지원.
    """
    context = get_current_context()
    ti = context["ti"]
    api_status = ti.xcom_pull(task_ids="get_branch_by_api_status")

    if api_status == "deploy_new_model":
        return "deploy_new_model"

    deployed_creation_time = ti.xcom_pull(
        task_ids="get_deployed_model_creation_time"
    )
    trained_creation_time = ti.xcom_pull(
        task_ids="get_latest_trained_model_creation_time"
    )

    print("deployed_creation_time", deployed_creation_time)
    print("trained_creation_time", trained_creation_time)

    if deployed_creation_time is None:
        print("There is no deployed model!")
        return "deploy_new_model"

    if (
        trained_creation_time is not None
        and trained_creation_time > deployed_creation_time
    ):
        print("Deployed model is already out-of-date.")
        return "deploy_new_model"

    print("Skip deployment.")
    return "skip_deployment"


with DAG(
    dag_id="credit_score_classification_cd",
    default_args={
        "owner": "user",
        "depends_on_past": False,
        "email": ["otzslayer@gmail.com"],
        "on_failure_callback": failure_callback,
        "on_success_callback": success_callback,
    },
    description="A DAG for continuous deployment",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=local_timezone),
    catchup=False,
    tags=set(["lgcns", "mlops"]),
) as dag:
    # TODO: API 상태 체크 결과 가져오기
    get_api_status_task = EmptyOperator(task_id="get_branch_by_api_status")

    # TODO: 현재 컨테이너에서 실행 중인 모델의 creation_time 가져오기
    get_deployed_model_creation_time_task = EmptyOperator(
        task_id="get_deployed_model_creation_time",
    )

    # TODO: 로컬에서 최신 학습된 모델의 creation_time 가져오기
    get_latest_trained_model_creation_time_task = EmptyOperator(
        task_id="get_latest_trained_model_creation_time",
    )

    # TODO: 모델을 업데이트할지 결정
    decide_update_task = EmptyOperator(
        task_id="decide_update",
    )

    # TODO: 새로운 모델을 배포
    deploy_new_model_task = EmptyOperator(
        task_id="deploy_new_model",
    )

    # 배포를 건너뛸 경우 실행할 더미 태스크
    skip_deployment_task = PythonOperator(
        task_id="skip_deployment",
        python_callable=lambda: print("No new model to deploy"),
    )

    # DAG 실행 순서 정의
    # 1️⃣ API가 정상 동작하지 않으면 즉시 배포
    get_api_status_task >> deploy_new_model_task

    # 2️⃣ API가 정상 동작하면 모델 생성 시간 비교 후 업데이트 결정
    (
        get_api_status_task
        >> [
            get_deployed_model_creation_time_task,
            get_latest_trained_model_creation_time_task,
        ]
        >> decide_update_task
    )

    # 3️⃣ decide_update_task의 결과에 따라 모델 배포 여부 결정
    decide_update_task >> [deploy_new_model_task, skip_deployment_task]
