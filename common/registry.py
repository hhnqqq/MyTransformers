import os
from typing import Union, List

class Registry:
    mapping = {
        "model_mapping":{},
        "pipeline_model_mapping":{},
        "train_model_mapping":{},
        "model_config_mapping":{},
        "dataset_mapping":{},
        "info_manager_mapping":{},
        "tokenizer_mapping":{},
        "paths_mapping":{}
    }

    @classmethod
    def register_path(cls, name, path):
        if name in cls.mapping['paths_mapping']:
            raise KeyError(
                "Name '{}' already registered for {}.".format(
                    name, cls.mapping["paths_mapping"][name]
                )
            )
        cls.mapping['paths_mapping'][name] = path


    @classmethod
    def register_dataset(cls, name):

        def warp(dataset_cls):
            # if name in cls.mapping['dataset_mapping']:
            #     raise KeyError(
            #         "Name '{}' already registered for {}.".format(
            #             name, cls.mapping["dataset_mapping"][name]
            #         )
            #     )            
            cls.mapping['dataset_mapping'][name] = dataset_cls
            return dataset_cls
        return warp

    @classmethod
    def register_info_manager(cls, name):

        def wrap(func):
            if name in cls.mapping['info_manager_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["info_manager_mapping"][name]
                    )
                )
            cls.mapping['info_manager_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_model(cls, model_names: Union[List[str], str]):
        if isinstance(model_names, str):
            model_names = [model_names]
        def wrap(model_cls):
            for model_name in model_names:
                if model_name in cls.mapping['model_mapping']:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            model_name, cls.mapping["model_mapping"][model_name]
                        )
                    )
                cls.mapping['model_mapping'][model_name] = model_cls
            return model_cls
        return wrap
    
    @classmethod
    def register_pipeline_model(cls, model_names: Union[List[str], str]):
        if isinstance(model_names, str):
            model_names = [model_names]
        def wrap(model_cls):
            for model_name in model_names:
                if model_name in cls.mapping['pipeline_model_mapping']:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            model_name, cls.mapping["pipeline_model_mapping"][model_name]
                        )
                    )
                cls.mapping['pipeline_model_mapping'][model_name] = model_cls
            return model_cls
        return wrap

    @classmethod
    def register_train_model(cls, model_names: Union[List[str], str]):
        if isinstance(model_names, str):
            model_names = [model_names]

        def wrap(model_cls):
            for model_name in model_names:
                # assert(issubclass(model_cls, BaseModel))
                if model_cls in cls.mapping['model_mapping']:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            model_name, cls.mapping["train_model_mapping"][model_name]
                        )
                    )
                cls.mapping['train_model_mapping'][model_name] = model_cls
            return model_cls
        return wrap


    @classmethod
    def register_model_config(cls, model_names: Union[List[str], str]):
        if isinstance(model_names, str):
            model_names = [model_names]

        def wrap(model_cls):
            for model_name in model_names:
                if model_name in cls.mapping['model_mapping']:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            model_name, cls.mapping["model_config_mapping"][model_name]
                        )
                    )
                cls.mapping['model_config_mapping'][model_name] = model_cls
            return model_cls
        return wrap
    
    @classmethod
    def register_info_manager(cls, name):

        def wrap(func):
            if name in cls.mapping['info_manager_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["info_manager_mapping"][name]
                    )
                )
            cls.mapping['info_manager_mapping'][name] = func
            return func
        return wrap
    
    @classmethod
    def register_tokenizer(cls, name):

        def wrap(tokenizer_cls):
            if name in cls.mapping['tokenizer_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["tokenizer_mapping"][name]
                    )
                )
            cls.mapping['tokenizer_mapping'][name] = tokenizer_cls
            return tokenizer_cls
        return wrap
    
    @classmethod
    def get_model_class(cls, name):
        result = cls.mapping["model_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name: {name} in model mapping, \
supported models are listed below:{cls.list_models()}")
        else:
            return result
    
    @classmethod
    def get_pipeline_model_class(cls, name):
        result = cls.mapping["pipeline_model_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name: {name} in pipeline model mapping, \
supported pipeline models are listed below:{cls.list_pipeline_models()}")
        else:
            return result
    
    @classmethod
    def get_model_config_class(cls, name):
        result = cls.mapping["model_config_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name: {name} in model config mapping, \
supported model configs are listed below:{cls.list_model_configs()}")
        else:
            return result
    
    @classmethod
    def get_train_model_class(cls, name):
        result = cls.mapping["train_model_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name: {name} in train model mapping, \
supported train models are listed below:{cls.list_train_models()}")
        else:
            return result
    
    @classmethod
    def get_tokenizer_class(cls, name):
        result =  cls.mapping["tokenizer_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name:{name} in tokenizer mapping, \
supported tokenizer are listed below:{cls.list_tokenizers()}")
        else:
            return result

    @classmethod
    def get_dataset_class(cls, name):
        result =  cls.mapping["dataset_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name:{name} in dataset mapping, \
supported dataset are listed below:{cls.list_datasets()}")
        else:
            return result
    
    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths_mapping"].get(name, None)
    
    @classmethod
    def get_paths(cls, args):
        """
        Get paths of model, tokenizer and dataset from `paths.json` into args.
        By utilizing this function, you can acquire the path by name.

        Examples:
            args.model_name == llama3 ---> args.ckpt_path == `path_to_llama3_model` 

            args.model_name == llama3 ---> args.tokenizer_path == `path_to_llama3_tokenizer` 

            args.model_name == llama3 ---> args.model_name_or_path == `path_to_llama3_hf_model` 
            
            args.train_dataset_name == gsm8k ---> args.train_dataset_path == `path_to_gsm8k`
        """
        # by doing this you are not need to provide paths in your script
        paths_mapping = cls.mapping["paths_mapping"]
        for k,v in cls.mapping["paths_mapping"].items():
            # Ensure the paths in mapping exist
            if isinstance(v, str) and not os.path.exists(v):
                paths_mapping[k] = None
            elif isinstance(v, list) and not (all([os.path.exists(item) for item in v])):
                paths_mapping[k] = None

        # Acquire model and tokenizer checkpoint paths first.
        # If use huggingface, then we only need model_name_or_path to acquire model and tokenizer.
        model_name = '_'.join([args.model_name, args.variant]) if (args.model_name and args.variant) else None

        if args.huggingface:
            if not (model_name or args.model_name_or_path):
                raise ValueError(
                    "Please provide either (args.model_name and args.variant) "
                    "or args.model_name_or_path for Hugging Face models."
                )

            if not args.model_name_or_path:
                hf_key = f"huggingface_{model_name}"
                args.model_name_or_path = paths_mapping.get(hf_key)

            if args.model_name_or_path is None:
                hf_mappings = {k: v for k, v in paths_mapping.items() if k.startswith('huggingface_')}
                raise ValueError(
                    f"Could not resolve model path for Hugging Face model '{model_name}'. "
                    f"Key 'huggingface_{model_name}' not found in paths_mapping. "
                    f"Available Hugging Face mappings: {list(hf_mappings.keys())}"
                )
        else:
            if not model_name:
                raise ValueError('Please provide (args.model_name and args.variant) for local pytorch models.')
            tokenizer_key = f"tokenizer_{args.model_name}"
            model_key = f"model_{model_name}"
            args.tokenizer_path = args.tokenizer_path or paths_mapping.get(tokenizer_key)
            args.ckpt_path = args.ckpt_path or paths_mapping.get(model_key)

        # Acquire the dataset file paths.
        train_dataset_name = "train_dataset_" + args.train_dataset_name
        eval_dataset_name = '' if args.eval_dataset_name is None else args.eval_dataset_name
        eval_dataset_name = "eval_dataset_" + eval_dataset_name
        args.train_dataset_path = args.train_dataset_path if args.train_dataset_path else paths_mapping.get(train_dataset_name, None)
        args.eval_dataset_path = args.eval_dataset_path if args.eval_dataset_path else paths_mapping.get(eval_dataset_name, None)
        
        if args.train_dataset_path is None:
            raise ValueError(f"Can not find name:{train_dataset_name} in paths mapping,"
                              f"supported paths are listed below:{cls.list_paths()}")
        if args.eval_dataset_path is None and not args.skip_eval:
            raise ValueError(f"Can not find name:{eval_dataset_name} in paths mapping,"
                              f"supported paths are listed below:{cls.list_paths()}."
                              "For skipping evaluation, please set `args.skip_eval=True`")
        return args
    
    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_mapping"].keys())
    
    @classmethod
    def list_pipeline_models(cls):
        return sorted(cls.mapping["pipeline_model_mapping"].keys())

    @classmethod
    def list_model_configs(cls):
        return sorted(cls.mapping["model_config_mapping"].keys())
    
    @classmethod
    def list_train_models(cls):
        return sorted(cls.mapping["train_model_mapping"].keys())

    @classmethod
    def list_paths(cls):
        return sorted(cls.mapping["paths_mapping"].keys())
    
    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["dataset_mapping"].keys())
    
    @classmethod
    def list_info_managers(cls):
        return sorted(cls.mapping["info_manager_mapping"].keys())
    
    @classmethod
    def list_tokenizers(cls):
        return sorted(cls.mapping["tokenizer_mapping"].keys())

    @classmethod
    def list_all(cls):
        return {k:sorted(v.keys()) for k,v in cls.mapping.items()}

    
registry = Registry()