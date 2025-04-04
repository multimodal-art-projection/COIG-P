cd ./Chinese_Reward_Model
conda activate KOR_Bench

python evaluator.py --score_pattern Discriminative --model_name Skywork-Reward-Gemma-2-27B-v0.2  
python evaluator.py --score_pattern Discriminative --model_name Llama-3-OffsetBias-RM-8B  
python evaluator.py --score_pattern Discriminative --model_name RM-Mistral-7B  
python evaluator.py --score_pattern Discriminative --model_name ArmoRM-Llama3-8B-v0.1 
python evaluator.py --score_pattern Discriminative --model_name Skywork-Reward-Llama-3.1-8B-v0.2

python calculate.py --model_name Skywork-Reward-Gemma-2-27B-v0.2  
python calculate.py --model_name Llama-3-OffsetBias-RM-8B    
python calculate.py --model_name RM-Mistral-7B   
python calculate.py --model_name ArmoRM-Llama3-8B-v0.1 
python calculate.py --model_name Skywork-Reward-Llama-3.1-8B-v0.2
python calculate.py --model_name CRM