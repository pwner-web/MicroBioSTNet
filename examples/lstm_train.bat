@echo off

cd ..
REM Start process for subject A Saliva
echo start at %date% %time% > time.stat
python main.py -i data/A_Saliva_abun.csv -s "subject A Saliva" --num_timesteps_output 1 --enable-cuda --lstm

REM Start process for subject A Stool
python main.py -i data/A_Stool_abun.csv -s "subject A Stool" --num_timesteps_output 1 --enable-cuda --lstm

REM Start process for subject M3 Stool
python main.py -i data/M3_gut_abun.csv -s "subject M3 Stool" --num_timesteps_output 1 --enable-cuda --lstm

REM Start process for subject M3 Saliva
python main.py -i data/M3_tongue_abun.csv -s "subject M3 Saliva" --num_timesteps_output 1 --enable-cuda --lstm

REM Start process for subject B Stool
python main.py -i data/B_Stool_abun.csv -s "subject B Stool" --num_timesteps_output 1 --enable-cuda --lstm

REM Start process for subject F4 Stool
python main.py -i data/F4_gut_abun.csv -s "subject F4 Stool" --num_timesteps_output 1 --enable-cuda --lstm

REM Start process for subject F4 Saliva
python main.py -i data/F4_tongue_abun.csv -s "subject F4 Saliva" --num_timesteps_output 1 --enable-cuda --lstm