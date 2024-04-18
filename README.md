# ChatGPT as an attack tool: Stealthy textual backdoor attack via blackbox generative model trigger 
 

**1. Poison dataset**

  Step 1: Poison dataset generation via ```script/astep1-ChatGPT_poison.sh```, you can place the clean dataset under the dataset/folder

  You can download the dataset from https://drive.google.com/file/d/1VnidCzAx7qJYIyNmP8EEJuyNxIxsdAH3/view?usp=sharing
  
  For the generation of style / Syntax, we used the code from https://github.com/INK-USC/BITE
  

**2. Backdoor Attack**
Step 3: You can conduct the backdoor attack via finetuning on BERT or LLaMA2 via ```sh astep-3-attack-BERT.sh``` or ```sh astep3-attack-LLM.sh```  You may need to first get the llama2 permission from Meta. 


**3. Stealthiness Evaluation**

Step 2: You check the stealthiness via ```sh astep2-stealthiness_eval.sh```.  The metrics include PPL, CoLA score, and GE score. 
        For sentiment consistence classification, we use 2-shot-in-context-learing. The demonstrations we used can be found in Appendix of paper. 






