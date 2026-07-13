import { NutritionData } from "../openai/extractNutrition";
import { PackagedItem } from "../openai/identifyPackagedItem";
export declare function lookupPackagedNutrition(item: PackagedItem): Promise<NutritionData | null>;
