class Regitry:
    mapping = {
        "model_mapping":{},
        "dataset_mapping":{},
        "info_manager_mapping":{},
        "tokenizer_mapping":{}
    }

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
    
registry = Regitry()