15-05-2025
Did a few changes will be mentioning them: 
GradientBoost Regeressor add kia models me because Gradient boosting model was not added 
then build the pipline engine do main files were added and one was a store supporter(to avoid store access loop):
1. trainer = where actual model training and all the metrics calculation it done
2. executer = it acts as bridge between the pipeline system and the trainer 
3. store = when running the swagger ui, the main file was stuck between two different run stores one was from pipline and one was from the engine so i shifted the run_store with the engine to avoid that conflict 

uske baad 
i edited YOR file pipeline.py to actually link the trainer with the executor 
ADDED A FAKE DATA CSV TOO for the pipeline testing purposes 

THE CHANGES ARE NOT COMPLETELY TESTED 
BECAUSE THE PIPELINE IS STIL FACING CACHED FILE ISSUESS IT FORGETS TO SAVE TO RUN ID which is needed to execute the pipeline 
SO IT IS RECOMMENNED NOT TO MERGE THE CHANGES ABHI I WILL FIX THAT PART TOMORROW THEN YOU CAN MERGE 
