# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import subprocess
import os
import logging
from jinja2 import Template
import re
import json

class PEMDXtb:
    def __init__(
            self,
            work_dir,
            chg=0,
            mult=1,
            gfn=2,
    ):
        self.work_dir = work_dir
        self.chg = chg
        self.mult = mult
        self.gfn = gfn

        # 创建工作目录（如果不存在）
        os.makedirs(self.work_dir, exist_ok=True)

        # 配置日志
        log_file = os.path.join(work_dir, 'xtb.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("PEMDXtb 实例已创建。")

    def run_local(self, xyz_filename, outfile_headname, optimize=True):

        uhf = self.mult - 1  # 未配对电子数

        # 定义命令模板
        command_template = Template(
            "xtb {{ xyz_filename }} {% if optimize %}--opt {% endif %} "
            "--chrg={{ chg }} "
            "--uhf={{ uhf }} "
            "--gfn {{ gfn }} "
            "--ceasefiles "
            "--namespace {{ work_dir }}/{{ outfile_headname }}"
        )

        # 渲染命令
        command_str = command_template.render(
            xyz_filename=xyz_filename,
            optimize=optimize,
            chg=self.chg,
            uhf=uhf,
            gfn=self.gfn,
            work_dir=self.work_dir,
            outfile_headname=outfile_headname
        )

        logging.info(f"执行命令: {command_str}")
        print(f"执行命令: {command_str}")

        try:
            result = subprocess.run(command_str, shell=True, check=True, text=True, capture_output=True)
            logging.info(f"XTB 输出: {result.stdout}")
            print(f"XTB 输出: {result.stdout}")

            # 提取能量信息
            energy_info = self.extract_energy(result.stdout, optimized=optimize)
            if energy_info:
                logging.info(f"提取的能量信息: {energy_info}")
                print(f"提取的能量信息: {energy_info}")
                # 保存能量信息到JSON文件
                energy_file = os.path.join(self.work_dir, f"{outfile_headname}_energy.json")
                with open(energy_file, 'w') as ef:
                    json.dump(energy_info, ef, indent=4)
                logging.info(f"能量信息已保存到 {energy_file}")
            else:
                logging.error("无法提取能量信息。")

            return {
                'output': result.stdout,
                'energy_info': energy_info,
                'error': None
            }

        except subprocess.CalledProcessError as e:
            logging.error(f"执行 XTB 时出错: {e.stderr}")
            print(f"执行 XTB 时出错: {e.stderr}")
            return {
                'output': None,
                'energy_info': None,
                'error': e.stderr
            }

    def extract_energy(self, stdout, optimized=False):

        energy_info = {}
        patterns = {
            'total_energy': r"::\s*total energy\s*([-+]?\d*\.\d+|\d+) Eh",
        }

        for key, pattern in patterns.items():
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                try:
                    if optimized:
                        energy = float(matches[-1])  # 取最后一个匹配
                    else:
                        energy = float(matches[0])   # 取第一个匹配
                    energy_info[key] = energy
                except ValueError:
                    logging.error(f"提取的能量值无法转换为浮点数: {matches[-1] if optimized else matches[0]}")
            else:
                logging.warning(f"未在输出中找到 '{key}' 信息。")

        if energy_info:
            return energy_info
        else:
            return None


