"""

"""

import os, shutil
from .ExtensionLoader import ExtensionLoader # copied from McUtils since I don't want a dependency here (yet)

__all__ = ["Config", "ConfigManager"]


class ConfigSerializer:

    loader = ExtensionLoader(rootpkg="Configs")

    @classmethod
    def get_serialization_mode(cls, file):
        _, ext = os.path.splitext(file)
        if ext == ".py":
            mode = "dict"
        elif ext == ".json":
            mode = "json"
        else:
            mode = "pickle"
        return mode

    @classmethod
    def serialize_dict(cls, file, ops):
        with open(file, "w+") as f:
            print(ops, f)
    @classmethod
    def serialize_module(cls, file, ops, attribute = "config"):
        with open(file, "w+") as f:
            f.write(attribute +" = ")
            print(ops, f)
    @classmethod
    def serialize_json(cls, file, ops):
        import json
        json.dump(ops, file)
    @classmethod
    def serialize_pickle(cls, file, ops):
        import pickle
        pickle.dump(ops, file)
    @classmethod
    def serialize(cls, file, ops, mode = None, attribute = None):
        if mode is None:
            mode = cls.get_serialization_mode(file)
        if mode == "dict":
            if attribute is not None:
                cls.serialize_module(file, ops, attribute = attribute)
            else:
                cls.serialize_dict(file, ops)
        elif mode == "json":
            cls.serialize_json(file, ops)
        elif mode == "pickle":
            cls.serialize_json(file, ops)
        else:
            raise ValueError("{}.{}: don't know serialization mode ''".format(
                cls.__name__,
                "serialize",
                mode
            ))

    @classmethod
    def deserialize_dict(cls, file):
        with open(file, "r") as f:
            return eval(f.read())
    @classmethod
    def deserialize_module(cls, file, attribute="config"):
        mod = cls.loader.load(file)
        return getattr(mod, attribute)
    @classmethod
    def deserialize_json(cls, file):
        import json
        return json.load(file)
    @classmethod
    def deserialize_pickle(cls, file):
        import pickle
        return pickle.load(file)
    @classmethod
    def deserialize(cls, file, mode = None, attribute = None):
        if mode is None:
            mode = cls.get_serialization_mode(file)
        if mode == "dict":
            if attribute is not None:
                cls.deserialize_module(file, attribute=attribute)
            else:
                cls.deserialize_dict(file)
        elif mode == "json":
            cls.deserialize_json(file)
        elif mode == "pickle":
            cls.deserialize_json(file)
        else:
            raise ValueError("{}.{}: don't know serialization mode ''".format(
                cls.__name__,
                "deserialize",
                mode
            ))

class Config:

    def __init__(self, config, root = None, loader = None):
        """Loads the config from a file

        :param loader:
        :type loader: ExtensionLoader | None
        :param config_file:
        :type config_file: str | dict | module
        """
        self.loader = loader
        self.root = root
        if isinstance(config, str) and not os.path.exists(os.path.abspath(config)):
            if isinstance(root, str):
                config = os.path.join(root, config)
            else:
                raise ConfigManagerError("Config file {} doesn't exist".format(config))
        self._conf = config
        self._conf_type = None
        self._conf_obj = None
        self._loaded = False

    @property
    def name(self):
        if self.root is not None:
            name = os.path.splitext(os.path.basename(self.root))[0]
        else:
            name = self.get_conf_attr("name")
        return name

    @property
    def conf(self):
        return self._conf

    @property
    def opt_dict(self):
        if self._conf_type is dict:
            return self._conf_obj
        else:
            return vars(self._conf_obj)

    def load_opts(self):
        if isinstance(self.conf, str):
            if ConfigSerializer.get_serialization_mode(str) == "dict":
                self._conf_mod = self.loader.load(self.conf) # why lose the reference?
                try:
                    self._conf_obj = self._conf_mod.config
                    self._conf_type = dict
                except AttributeError:
                    self._conf_obj = self._conf_mod
            else:
                self._conf_obj = ConfigSerializer.deserialize(str)
                self._conf_type = dict
        elif isinstance(self.conf, dict):
            self._conf_type = dict
            self._conf_obj = self.conf
        else:
            self._conf_obj = self.conf
        self._loaded = True

    def get_conf_attr(self, item):
        if not self._loaded:
            self.load_opts()
        if self._conf_type is dict:
            return self._conf_obj[item]
        else:
            return getattr(self._conf_obj, item)

    def __getattr__(self, item):
        return self.get_conf_attr(item)

    def get_file(self, name, conf_attr = "root"):
        root = self.root
        if root is None:
            root = self.get_conf_attr("root")
        return os.path.join(root, name)

class ConfigManager:
    """
    Manages configurations inside a single directory
    """

    def __init__(self, conf_dir, conf_file = "config.py", config_package = "Configs"):
        self.conf_dir = conf_dir
        self.conf_name = conf_file
        self.loader = ExtensionLoader(rootdir = self.conf_dir, rootpkg = config_package)
        if not os.path.isdir(conf_dir):
            os.mkdir(conf_dir)

    def list_configs(self):
        """Lists the configurations known by the ConfigManager

        :return: the names of the configs stored in the config directory
        :rtype: List[str]
        """
        return [
            Config(self.conf_name, root = os.path.join(self.conf_dir, r)).name for r in os.listdir(self.conf_dir) if (
                os.path.isdir(os.path.join(self.conf_dir, r))
            )
        ]

    def config_loc(self, name):
        return os.path.join(self.conf_dir, name)

    def load_config(self, name, conf_file = None):
        """Loads a Config by name from the directory

        :param name: the name of the config file
        :type name: str
        :param conf_file: the config file to load from -- defaults to `'config.py'`
        :type conf_file: str | None
        :return:
        :rtype:
        """
        if conf_file is None:
            conf_file = self.conf_name
        return Config( conf_file,  root=os.path.join(self.conf_dir, name), loader=self.loader )

    def add_config(self, name, config_file = None, **opts):
        """Adds a config to the known list. Requires a name. Takes either a config file or a bunch of options.

        :param name:
        :type name:
        :param conf_file:
        :type conf_file:
        :return:
        :rtype:
        """
        if os.path.isdir(os.path.join(self.conf_dir, name)):
            raise ConfigManagerError("A config with the name {} already exists in {}".format(
                name,
                self.conf_dir
            ))
        os.mkdir(os.path.join(self.conf_dir, name))
        if config_file is not None:
            shutil.copy(config_file, os.path.join(self.conf_dir, name, self.conf_name))
        else:
            ConfigSerializer.serialize(
                os.path.join(self.conf_dir, name, self.conf_name),
                opts,
                attribute="config"
            )

class ConfigManagerError(Exception):
    pass
