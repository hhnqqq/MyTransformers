import os
class Regitry:
    mapping = {
        "model_mapping":{},
        "train_model_mapping":{},
        "model_config_mapping":{},
        "dataset_mapping":{},
        "info_manager_mapping":{},
        "tokenizer_mapping":{},
        "paths_mapping":{}
    }

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
    def register_model(cls, name, model_cls):
        if model_cls in cls.mapping['model_mapping']:
            raise KeyError(
                "Name '{}' already registered for {}.".format(
                    model_cls, cls.mapping["model_mapping"][name]
                )
            )
        cls.mapping['model_mapping'][name] = model_cls

    @classmethod
    def register_model_config(cls, name, model_cls):
        if model_cls in cls.mapping['model_mapping']:
            raise KeyError(
                "Name '{}' already registered for {}.".format(
                    model_cls, cls.mapping["model_config_mapping"][name]
                )
            )
        cls.mapping['model_config_mapping'][name] = model_cls

    @classmethod
    def register_train_model(cls, name, model_cls):
        if model_cls in cls.mapping['model_mapping']:
            raise KeyError(
                "Name '{}' already registered for {}.".format(
                    model_cls, cls.mapping["train_model_mapping"][name]
                )
            )
        cls.mapping['train_model_mapping'][name] = model_cls

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
        return cls.mapping["model_mapping"].get(name, None)
    
    @classmethod
    def get_model_config_class(cls, name):
        return cls.mapping["model_config_mapping"].get(name, None)
    
    @classmethod
    def get_train_model_class(cls, name):
        return cls.mapping["train_model_mapping"].get(name, None)
    
    @classmethod
    def get_tokenizer_class(cls, name):
        return cls.mapping["tokenizer_mapping"].get(name, None)
    
    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths_mapping"].get(name, None)
    
    @classmethod
    def get_paths(cls, args):
        # by doing this you are not need to provide paths in your script
        paths_mapping = cls.mapping["paths_mapping"]
        for k,v in cls.mapping["paths_mapping"].items():
            if not os.path.isfile(v):
                paths_mapping[k] = None
        tokenizer_name = "tokenizer_" + args.model_name
        model_name = "model_"  + args.model_name
        dataset_name = "dataset_" + args.dataset_name
        args.tokenizer_path = args.tokenizer_path if args.tokenizer_path else paths_mapping.get(tokenizer_name, None)
        args.dataset_path = args.dataset_path if args.dataset_path else paths_mapping.get(dataset_name, None)
        args.ckptl_path = args.ckpt_path if args.ckpt_path else paths_mapping.get(model_name, None)
        return args
    
    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_mapping"].keys())

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
        return sorted(cls.mapping["datasets_mapping"].keys())
    
    @classmethod
    def list_info_managers(cls):
        return sorted(cls.mapping["info_manager_mapping"].keys())
    
    @classmethod
    def list_info_tokenizers(cls):
        return sorted(cls.mapping["tokenizer_mapping"].keys())
    
registry = Regitry()