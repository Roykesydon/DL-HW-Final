from domain.virtual_environment import VirtualEnvironment
from domain.decision_module import DecisionModule
from torch import tensor
from configs.simulate_config import SIMULATE_CONFIG

result_folder_name = SIMULATE_CONFIG["result_folder_name"]
TRAIN_END_DATE = SIMULATE_CONFIG["train_end_date"]

decision_module = DecisionModule()

decision_module.load_model(f"./results/{result_folder_name}/weight.pth")
decision_module.set_mean_and_std(
    SIMULATE_CONFIG["mean_and_std"]
)

virtual_environment = VirtualEnvironment(train_end_date=TRAIN_END_DATE)
virtual_environment.set_decision_module(decision_module)

virtual_environment.run()
