"""

"""

import os, shutil
from .ModuleLoader import ModuleLoader # copied from McUtils since I don't want a dependency here (yet)

__all__ = ["Config", "ConfigManager", "ConfigSerializer"]


class ConfigSerializer:

    loader = ModuleLoader(rootpkg="Configs")

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
            print(ops, file=f)
    @classmethod
    def serialize_module(cls, file, ops, attribute = "config"):
        with open(file, "w+") as f:
            print(attribute, " = ", ops, file=f)
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
                return cls.deserialize_module(file, attribute=attribute)
            else:
                return cls.deserialize_dict(file)
        elif mode == "json":
            return cls.deserialize_json(file)
        elif mode == "pickle":
            return cls.deserialize_json(file)
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
        :type loader: ModuleLoader | None
        :param config_file:
        :type config_file: str | dict | module
        """
        self.loader = loader
        if isinstance(root, str):
            root = os.path.abspath(root)
        self.root = root
        if isinstance(config, str):
            abs_conf = os.path.abspath(config)
            rel_conf = os.path.abspath(config) != config
            if rel_conf:
                if isinstance(root, str):
                    config = os.path.join(root, config)
                elif os.path.exists(abs_conf):
                    config = abs_conf
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
        self.load_opts()
        if self._conf_type is dict:
            return self._conf_obj
        else:
            return vars(self._conf_obj)

    def update(self, **kw):
        opts = self.opt_dict
        opts.update(**kw)
        self._conf_type = dict
        self._conf_obj = opts

    def save(self, file=None, mode=None, attribute="config"):
        if file is None:
            file = self.conf
        if isinstance(file, str):
            ConfigSerializer.serialize(file, self.opt_dict, mode=mode, attribute=attribute)

    def load_opts(self):
        if not self._loaded:
            if isinstance(self.conf, str):
                if ConfigSerializer.get_serialization_mode(self.conf) == "dict":
                    if self.loader is None:
                        self.loader = ModuleLoader(rootpkg="Configs")
                    if not os.path.exists(self.conf):
                        raise IOError("Config file {} doesn't exist".format(self.conf))
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

    def get_file(self, name, conf_attr = "root"):
        root = self.root
        if root is None:
            root = self.get_conf_attr(conf_attr)
        return os.path.join(root, name)

    def get_conf_attr(self, item):
        if not self._loaded:
            self.load_opts()
        if self._conf_type is dict:
            return self._conf_obj[item]
        else:
            return getattr(self._conf_obj, item)

    def __getattr__(self, item):
        return self.get_conf_attr(item)
    # def __setattr__(self, key, value):
    #     self.update(**{key:value})

class ConfigManager:
    """
    Manages configurations inside a single directory
    """

    def __init__(self, conf_dir, conf_file = "config.py", config_package = "Configs"):
        if isinstance(conf_dir, str):
            conf_dir = os.path.abspath(conf_dir)
        self.conf_dir = conf_dir
        self.conf_name = conf_file
        self.loader = ModuleLoader(rootdir = self.conf_dir, rootpkg = config_package)
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

    def check_config(self, config_tag):
        return os.path.isdir(os.path.join(self.conf_dir, config_tag))

    def add_config(self, config_tag, config_file = None, **opts):
        """Adds a config to the known list. Requires a name. Takes either a config file or a bunch of options.

        :param name:
        :type name:
        :param conf_file:
        :type conf_file:
        :return:
        :rtype:
        """
        if os.path.isdir(os.path.join(self.conf_dir, config_tag)):
            raise ConfigManagerError("A config with the name {} already exists in {}".format(
                config_tag,
                self.conf_dir
            ))
        os.mkdir(os.path.join(self.conf_dir, config_tag))
        if config_file is not None:
            shutil.copy(config_file, os.path.join(self.conf_dir, config_tag, self.conf_name))
        else:
            ConfigSerializer.serialize(
                os.path.join(self.conf_dir, config_tag, self.conf_name),
                opts,
                attribute="config"
            )

    def remove_config(self, name):
        """
        Removes a config from the known list. Requires a name.

        :param name:
        :type name:
        :return:
        :rtype:
        """
        loc = self.config_loc(name)
        if os.path.isdir(loc):
            shutil.rmtree(loc)
        else:
            os.remove(loc)

    def edit_config(self, config_tag, **opts):
        """
        Updates a config from the known list. Requires a name.

        :param name:
        :type name:
        :return:
        :rtype:
        """

        conf = self.load_config(config_tag)
        conf.update(**opts)
        conf.save()

class ConfigManagerError(Exception):
    pass
