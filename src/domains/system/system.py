from typing import Type
import logging
from config import Config

from domains.model.info.isa_model_info import ISAModelInfo
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes.system_mode import SystemMode
from domains.visualizer.visualizers import Visualizer
from domains.visualizer.result_savers import ResultSaver


class System:
    def __init__(self, system_modes: list[SystemMode]) -> None:
        self.system_modes = system_modes

    def run(self) -> ISAModelInfoCollection:
        isa_model_info_list = []
        for system_mode in self.system_modes:
            result_collection = system_mode.run()
            isa_model_info = ISAModelInfo(system_mode.isa_model_configuration, result_collection)
            isa_model_info_list.append(isa_model_info)

        isa_model_info_collection = ISAModelInfoCollection(isa_model_info_list)
        isa_model_info_collection.print()
        return isa_model_info_collection

    def run_and_visualize(
        self, visualizers: list[Type[Visualizer]], result_savers: list[ResultSaver]
    ) -> None:
        logger = logging.getLogger()
        logger_level_original = logger.level
        logger.setLevel(max(logging.INFO, logger_level_original))

        isa_model_info_collection = self.run()

        for visualizer in visualizers:
            visualizer.visualize(isa_model_info_collection)
            print()

        if Config.RUN_RESULT_SAVERS:
            for result_saver in result_savers:
                result_saver.save_in_directory(isa_model_info_collection)

        logger.setLevel(logger_level_original)
