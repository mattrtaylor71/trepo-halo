import { z } from "zod";
export declare const FastNutritionSchema: z.ZodObject<{
    dish_name: z.ZodNullable<z.ZodString>;
    calories: z.ZodNullable<z.ZodNumber>;
    protein_g: z.ZodNullable<z.ZodNumber>;
    carbs_g: z.ZodNullable<z.ZodNumber>;
    fat_g: z.ZodNullable<z.ZodNumber>;
    confidence: z.ZodUnion<[z.ZodNumber, z.ZodEffects<z.ZodString, number, string>, z.ZodEffects<z.ZodObject<{}, "passthrough", z.ZodTypeAny, z.objectOutputType<{}, z.ZodTypeAny, "passthrough">, z.objectInputType<{}, z.ZodTypeAny, "passthrough">>, number, z.objectInputType<{}, z.ZodTypeAny, "passthrough">>, z.ZodEffects<z.ZodAny, number, any>]>;
    summary: z.ZodNullable<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    dish_name: string | null;
    calories: number | null;
    confidence: number;
    protein_g: number | null;
    carbs_g: number | null;
    fat_g: number | null;
    summary: string | null;
}, {
    dish_name: string | null;
    calories: number | null;
    protein_g: number | null;
    carbs_g: number | null;
    fat_g: number | null;
    summary: string | null;
    confidence?: any;
}>;
export type FastNutritionData = z.infer<typeof FastNutritionSchema>;
export declare function extractNutritionFast(imageBuffer: Buffer): Promise<FastNutritionData>;
