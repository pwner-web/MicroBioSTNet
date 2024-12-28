echo start at $(date +%Y-%m-%d" "%H:%M:%S) && \
cd .. && \
python main.py -i data/A_Saliva_abun.csv -s "subject A Saliva" --num_timesteps_output 1 && \
echo subject A Saliva end at $(date +%Y-%m-%d" "%H:%M:%S) && \
python main.py -i data/A_Stool_abun.csv -s "subject A Stool" --num_timesteps_output 1 && \
echo subject A Stool end at $(date +%Y-%m-%d" "%H:%M:%S) && \
python main.py -i data/M3_gut_abun.csv -s "subject M3 Stool" --num_timesteps_output 1 && \
echo subject M3 Stool end at $(date +%Y-%m-%d" "%H:%M:%S) && \
python main.py -i data/M3_tongue_abun.csv -s "subject M3 Saliva" --num_timesteps_output 1 && \
echo subject M3 Saliva end at $(date +%Y-%m-%d" "%H:%M:%S) && \
python main.py -i data/B_Stool_abun.csv -s "subject B Stool" --num_timesteps_output 1 && \
echo subject B Stool end at $(date +%Y-%m-%d" "%H:%M:%S) && \
python main.py -i data/F4_gut_abun.csv -s "subject F4 Stool" --num_timesteps_output 1 && \
echo subject F4 Stool end at $(date +%Y-%m-%d" "%H:%M:%S) && \
python main.py -i data/F4_tongue_abun.csv -s "subject F4 Saliva" --num_timesteps_output 1 && \
echo subject F4 Saliva end at $(date +%Y-%m-%d" "%H:%M:%S)