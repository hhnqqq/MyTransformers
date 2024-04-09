class Regitry:
    mapping = {
        "model_mapping":{},
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
                    name, cls.mapping["ipaths_mapping"][name]
                )
            )
        cls.mapping['paths_mapping'][name] = path


    @classmethod
    def register_model(cls, name):
        # from model import BaseModel
        pass

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
    def get_tokenizer_class(cls, name):
        return cls.mapping["tokenizer_mapping"].get(name, None)
    
    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths_mapping"].get(name, None)
    
    @classmethod
    def get_paths(cls, args):
        # by doing this you are not need to provide paths in your script
        paths_mapping = cls.mapping["paths_mapping"]
        tokenizer_name = "tokenizer_" + args.model_name
        model_name = "model_"  + args.model_name
        dataset_name = "dataset_" + args.dataset_name
        args.tokenizer_path = args.tokenizer_path if args.tokenizer_path else paths_mapping.get(tokenizer_name, None)
        args.dataset_path = args.dataset_path if args.dataset_path else paths_mapping.get(model_name, None)
        args.model_path = args.model_path if args.model_path else paths_mapping.get(dataset_name, None)
        return args
    
registry = Regitry()