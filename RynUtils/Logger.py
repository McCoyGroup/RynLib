import os

__all__ = [ "Logger" ]

class Logger:

    def __init__(self, log_file, verbosity = 0):
        self.log_file = log_file
        self.verbosity = verbosity

    def log_print(self, message, *params, **kwargs):
        if 'verbosity' in kwargs:
            verbosity = kwargs['verbosity']
            del kwargs['verbosity']
        else:
            verbosity = 0
        if verbosity <= self.verbosity and self.log_file is not None:
            log = self.log_file
            if isinstance(log, str):
                if not os.path.isdir(os.path.dirname(log)):
                    try:
                        os.makedirs(os.path.dirname(log))
                    except OSError:
                        pass
                #O_NONBLOCK is *nix only
                with open(log, "a", os.O_NONBLOCK) as lf: # this is potentially quite slow but I am also quite lazy
                    if len(params) > 0:
                        message = message.format(*params)
                    print(message, file = lf, **kwargs)
            else:
                if len(params) > 0:
                    message = message.format(*params)
                print(message, file = log, **kwargs)