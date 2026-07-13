import { z } from "zod";
export declare const PackagedItemSchema: z.ZodObject<{
    is_packaged_item: z.ZodBoolean;
    brand: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    product_name: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    variant: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    category: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    barcode: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    serving_size: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    confidence: z.ZodNumber;
    explanation: z.ZodOptional<z.ZodNullable<z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    confidence: number;
    is_packaged_item: boolean;
    serving_size?: string | null | undefined;
    explanation?: string | null | undefined;
    category?: string | null | undefined;
    brand?: string | null | undefined;
    product_name?: string | null | undefined;
    variant?: string | null | undefined;
    barcode?: string | null | undefined;
}, {
    confidence: number;
    is_packaged_item: boolean;
    serving_size?: string | null | undefined;
    explanation?: string | null | undefined;
    category?: string | null | undefined;
    brand?: string | null | undefined;
    product_name?: string | null | undefined;
    variant?: string | null | undefined;
    barcode?: string | null | undefined;
}>;
export type PackagedItem = z.infer<typeof PackagedItemSchema>;
export declare function identifyPackagedItem(imageBuffer: Buffer): Promise<PackagedItem>;
