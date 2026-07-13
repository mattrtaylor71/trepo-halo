import { z } from 'zod';
export declare const DishSchema: z.ZodObject<{
    dish_name: z.ZodNullable<z.ZodString>;
    category: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    cuisine_type: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    confidence: z.ZodUnion<[z.ZodNumber, z.ZodEffects<z.ZodString, number, string>, z.ZodEffects<z.ZodObject<{}, "passthrough", z.ZodTypeAny, z.objectOutputType<{}, z.ZodTypeAny, "passthrough">, z.objectInputType<{}, z.ZodTypeAny, "passthrough">>, number, z.objectInputType<{}, z.ZodTypeAny, "passthrough">>, z.ZodEffects<z.ZodAny, number, any>]>;
    explanation: z.ZodOptional<z.ZodNullable<z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    dish_name: string | null;
    confidence: number;
    explanation?: string | null | undefined;
    category?: string | null | undefined;
    cuisine_type?: string | null | undefined;
}, {
    dish_name: string | null;
    confidence?: any;
    explanation?: string | null | undefined;
    category?: string | null | undefined;
    cuisine_type?: string | null | undefined;
}>;
export type Dish = z.infer<typeof DishSchema>;
/** When recharacterizing, pass current row state so the model preserves previous corrections and only applies the new one. */
export interface ExistingDishContext {
    dish_name?: string | null;
    explanation?: string | null;
}
export declare function identifyDish(imageBuffer: Buffer, userCorrection?: string | null, existingContext?: ExistingDishContext | null): Promise<Dish>;
