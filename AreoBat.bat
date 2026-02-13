@echo off
:: 1. 切换到当前脚本所在的目录（确保路径是对的）
cd /d %~dp0

:: 2. 激活你的 aerobat 虚拟环境
:: 注意：这是根据你之前报错信息推断的路径，如果不对请手动修改
call C:\Users\27626\anaconda3\Scripts\activate.bat aerobat

:: 3. 打印提示信息
echo ==========================================
echo       正在启动 AeroBat 仿真平台...
echo ==========================================

:: 4. 运行 Python 主程序
python main.py

:: 5. 如果程序崩溃或结束，暂停窗口让你看报错（程序正常关闭后这行也会执行）
pause