import { NutritionData } from '../openai/extractNutrition';
import { Dish } from '../openai/identifyDish';
/**
 * Generates a realistic standardized food image using gpt-image-1
 * Uses all available dish context for accurate, detailed images
 * gpt-image-1 is OpenAI's best image generation model with highest quality and prompt adherence
 * Returns the image as a Buffer
 */
export declare function generateDishImage(dish: Dish, nutritionData: NutritionData): Promise<Buffer | null>;
