# Revalidation Data Prep Scripts

This directory contains scripts for preparing the old algorithm results to be compared with the new algorithm results.

These scripts load the old results, recode the data, and save them in an easier to parse format.
The recoded data is saved in the `revalidation_data` repo and uploaded to Github: 
https://github.com/mobilise-d/mobgap_validation

To run the scripts, you need to follow these steps:

1. Clone mobgap_validation repo (https://github.com/mobilise-d/mobgap_validation)
2. Get the old algorithm results from the SUSTAIN Sharepoint. Only project members have access to this data.
   (https://ucd.sharepoint.com/:f:/r/sites/SUSTAIN-Mobilise-D/Shared%20Documents/Working%20Goups/Mobgap%20Validation/eScience/Block%20by%20block%20results?csf=1&web=1&e=eTAdVI)
3. Download these results. 
   You can store them anywhere on your computer.
   However, if you don't want to update the path in the scripts, place them in a subfolder called `_old_data_raw` within the cloned `revalidation_data` repo.
   UNDER NO CIRCUMSTANCES SHOULD YOU UPLOAD THE OLD DATA TO GITHUB!
4. Set your environment variables.
   `MOBGAP_VALIDATION_DATA_PATH` should point to the `revalidation_data` repo.
   The easiest way to do that is to add a `.env` file to the root of your local `mobgap` repo.

Now you can run the scripts.

## After updating results

When you modified and rerun scripts, use git to double-check the changes made to the revalidation data.
If you are happy, commit and push the changes to the `revalidation_data` repo.

Make sure you also commit and push any changes to the scripts in this repo.

