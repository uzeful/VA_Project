import yaml
"""
This is the config class used for configuring the programs like 
neural networks or just the normal programs, with the yaml config file
usage:
        from config_tools import Config
        from optparse import OptionParser
        parser = OptionParser()
        parser.add_option('--config',
                          type=str,
                          help="net configuration",
                          default="example.yaml")
        
        def main(argv):
            (opts, args) = parser.parse_args(argv)
            assert isinstance(opts, object)
            config = Config(opts.config)
            print(config)
----------------------------------------------------------
        The yaml file looks like:

        train:
            max_iter: 5000
            display: 10
            snapshot_iter: 1000
            batch_size: 64
            latent_dims: 100
            log: ../outputs/mnist_gan/mnist_gan.log
            snapshot_prefix: ../outputs/mnist_gan/mnist_gan
            scale: 2.0
            bias: 0.5
"""

class Config(object):
    def __init__(self, config):
        stream = open(config,'r')
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                cmd = "self." + k + "=" + repr(v)
                print(cmd)
                exec(cmd)

        stream.close()
