from types import FunctionType



# class DatasetConfig(BaseConfig):
#     statistics_config: StatisticsConfig = StatisticsConfig()
#     features_config: FeaturesConfig = FeaturesConfig()


# class InferenceConfig(BaseConfig):
#     BEAM_WIDTH = 5  # int | None : The beam width.
#     # Inputting None or int greater than number of tags results in standard Viterbi

#     USE_MULTI_CORE = True  # bool:  Whether to utilize a multi-core compute over each sentence
#     PARALLEL_POOL_SIZE = 4  # cpu_count(logical=False)  # Number of processes to use in the parallel pool
#     DISP_STATUS_EVERY = 25  # Display a status for every process every DISP_STATUS_EVERY sentences. Use None to shut
#     MAX_CONFUSION_MAT_SIZE = None  # Maximum matrix dimension to display. Use None to plot the entire matrix
#     PRED_DIR: Path = (Path(__file__).parents[2] / 'predictions').resolve()


# class Config(BaseConfig):
#     # Filesystem
#     DATA_DIR = (Path(__file__).parents[2] / 'data').resolve()  # Path : Absolute path to data dir

#     # Output to screen
#     VERBOSE_LEVEL = 1  # int in {0,1,2} : 0 for no output, 1 for headlines, 2 for everything
#     RANDOM_SEED = 7
#     # Dataset Handling
#     dataset_config: DatasetConfig = DatasetConfig()
#     train_config: TrainConfig = TrainConfig()
#     infer_config: InferenceConfig = InferenceConfig()



# In a different file due to circular dependencies
class BaseConfig:
    def __init__(self, **kwargs):  # Only allow init via kwargs
        self.__dict__.update(**kwargs)

    def __str__(self, indent_level=1):
        rep = f"{self.__class__.__name__}:\n"
        mems = []
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_') and not isinstance(v, (staticmethod, classmethod, FunctionType)):
                padding = '\t' * indent_level
                if isinstance(v, BaseConfig):
                    if k in self.__dict__:
                        mems.append(f'{padding}{k} : {self.__dict__[k].__str__(indent_level=indent_level + 1)}')
                    else:
                        mems.append(f'{padding}{k} : {v.__str__(indent_level=indent_level + 1)}')
                else:
                    if k in self.__dict__:
                        mems.append(f'{padding}{k} : {self.__dict__[k]}')
                    else:
                        mems.append(f'{padding}{k} : {v}')
        rep += "\n".join(mems)
        return rep

    @classmethod
    def assimilate(cls, name_dict):
        relevant_dict = {k: v for k, v in name_dict.items() if k in cls.__dict__}
        new_cfg = cls(**relevant_dict)
        # Now override the existing paramters:
        for k, v in cls.__dict__.items():
            if isinstance(v, BaseConfig):
                new_cfg.__dict__[k] = v.assimilate(name_dict)
        return new_cfg
