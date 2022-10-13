import spdlog as spd
from pathlib import Path
import datetime
import os


# Logger singelton
class RailTwinLogger:
    # fix for older spd version
    if not os.name == "nt":
        _logger_generators = {"console": lambda: spd.stderr_color_sink_mt()}
    else:
        _logger_generators = {"console": lambda: spd.stdout_sink_mt()}
    instance = None

    def __init__(self):
        self.sinks = {}
        for name, generator in RailTwinLogger._logger_generators.items():
            self.sinks[name] = generator()
        self.logger = spd.SinkLogger("RT-Logger", list(self.sinks.values()))
        self.logger.flush()

    def __del__(self):
        self.logger.flush()

    @staticmethod
    def register_file_logger(path=Path("")):
        if path.exists() and path.is_dir():
            _logger_output_path = path
        else:
            _logger_output_path = Path(__file__).parent.parent.resolve() / "log"
            RailTwinLogger.logger.info("Default Path selected")

        RailTwinLogger._logger_generators["file"] = lambda: spd.FileLogger("RT-FileLogger", _logger_output_path /
                                                                           "{}-{}".format(
                                                                               datetime.today().strftime('%Y-%m-%d'),
                                                                               "RailTwin_Reference_Run"))

    @staticmethod
    def create():
        if not RailTwinLogger.instance:
            RailTwinLogger.instance = RailTwinLogger()
        return RailTwinLogger.instance.logger
