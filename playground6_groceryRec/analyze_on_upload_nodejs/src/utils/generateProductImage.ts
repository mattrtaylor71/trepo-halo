import OpenAI from 'openai';

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY environment variable is required');
    }
    client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }
  return client;
}

import { GroceryItem } from '../openai/identifyGrocery';

/**
 * Generates a beautiful comic animation style image of a grocery product using gpt-image-1
 * Uses all available product context for accurate, detailed images
 * gpt-image-1 is OpenAI's best image generation model with highest quality and prompt adherence
 * Returns the image as a Buffer
 */
export async function generateProductWireframe(
  groceryItem: GroceryItem
): Promise<Buffer | null> {
  const openai = getClient();

  // Build comprehensive product description using all available context
  const parts: string[] = [];
  
  // Brand and product name
  if (groceryItem.brand && groceryItem.product_name) {
    parts.push(`${groceryItem.brand} ${groceryItem.product_name}`);
  } else if (groceryItem.product_name) {
    parts.push(groceryItem.product_name);
  } else if (groceryItem.brand) {
    parts.push(groceryItem.brand);
  }
  
  // Variant (size, flavor, etc.)
  if (groceryItem.variant) {
    parts.push(groceryItem.variant);
  }
  
  // Category for context
  if (groceryItem.category) {
    parts.push(`(${groceryItem.category})`);
  }
  
  const productDescription = parts.join(' ') || 'grocery product';
  
  // Build additional context hints from ingredients/nutrition
  const contextHints: string[] = [];
  
  if (groceryItem.ingredients && groceryItem.ingredients.length > 0) {
    // Extract key visual characteristics from ingredients
    const ingredientsStr = groceryItem.ingredients.join(', ').toLowerCase();
    if (ingredientsStr.includes('chocolate')) contextHints.push('chocolate');
    if (ingredientsStr.includes('strawberry') || ingredientsStr.includes('berry')) contextHints.push('berry');
    if (ingredientsStr.includes('vanilla')) contextHints.push('vanilla');
    if (ingredientsStr.includes('mint')) contextHints.push('mint');
    if (ingredientsStr.includes('lemon') || ingredientsStr.includes('citrus')) contextHints.push('citrus');
    if (ingredientsStr.includes('apple')) contextHints.push('apple');
    if (ingredientsStr.includes('banana')) contextHints.push('banana');
    if (ingredientsStr.includes('orange')) contextHints.push('orange');
  }
  
  // Build detailed prompt for comic animation style
  let prompt = `A beautiful comic animation style illustration of ${productDescription}`;
  
  // Add visual context hints
  if (contextHints.length > 0) {
    prompt += ` with ${contextHints.join(' and ')} characteristics`;
  }
  
  prompt += `. 
  
Style: Vibrant comic book animation style, like a high-quality animated movie or comic book illustration. 
Bright, saturated colors. Clean lines. Professional animation quality.
The product should be the main focus, centered and clearly visible.
Accurate representation of the product's actual appearance, packaging, and characteristics.
Show the product in an appealing, appetizing way that makes it look delicious and attractive.
Background should be simple and clean, either transparent or a subtle gradient that complements the product.
NO TEXT. NO WORDS. NO LABELS. NO BRAND NAMES VISIBLE. NO BARCODES. NO WRITING OF ANY KIND.
Just the beautiful product illustration in comic animation style.
High detail, professional quality, suitable for use in a premium grocery app.`;

  try {
    console.log('[generateImage] Generating comic animation style image for:', productDescription);
    if (contextHints.length > 0) {
      console.log('[generateImage] Context hints:', contextHints.join(', '));
    }
    
    // Use gpt-image-1 - OpenAI's best image generation model
    // Optimized for app performance: JPEG format with compression, medium quality
    // According to OpenAI docs, gpt-image-1 returns base64-encoded images in b64_json
    const response = await openai.images.generate({
      model: 'gpt-image-1',
      prompt: prompt,
      size: '1024x1024', // Smallest available size for gpt-image-1
      quality: 'medium', // Medium quality - good balance of quality and file size
      output_format: 'jpeg', // JPEG is much smaller than PNG and loads faster
      output_compression: 75, // 75% compression - good quality with smaller file size
      n: 1,
    });

    // gpt-image-1 response structure: { created, data: [{ b64_json, ... }] }
    // The image is returned as base64-encoded JSON, not a URL
    let imageBase64: string | undefined;
    
    if (response.data && Array.isArray(response.data) && response.data.length > 0) {
      imageBase64 = response.data[0]?.b64_json;
    }
    
    if (!imageBase64) {
      console.error('[generateImage] No base64 image data returned. Response structure:', JSON.stringify({
        hasData: !!response.data,
        dataLength: Array.isArray(response.data) ? response.data.length : 'not array',
        dataType: typeof response.data,
        keys: Object.keys(response || {}),
        firstDataItem: Array.isArray(response.data) && response.data.length > 0 ? Object.keys(response.data[0] || {}) : 'no items'
      }, null, 2));
      return null;
    }

    console.log('[generateImage] Image generated, decoding base64 data...');
    
    // Decode the base64 image data
    // Image is already optimized: JPEG format with 75% compression, medium quality
    // This provides significant size reduction compared to PNG/high quality
    const imageBuffer = Buffer.from(imageBase64, 'base64');
    console.log('[generateImage] Optimized JPEG image size:', imageBuffer.length, 'bytes');
    
    return imageBuffer;
  } catch (error) {
    console.error('[generateImage] Error generating product wireframe:', error);
    // Return null on error - non-fatal, we can continue without the wireframe
    return null;
  }
}

/**
 * Generates a minimal, flat icon-style image of a grocery product using gpt-image-1
 * Returns the image as a Buffer
 */
export async function generateProductIcon(
  groceryItem: GroceryItem
): Promise<Buffer | null> {
  const openai = getClient();

  // Build comprehensive product description using all available context
  const parts: string[] = [];
  
  // Brand and product name
  if (groceryItem.brand && groceryItem.product_name) {
    parts.push(`${groceryItem.brand} ${groceryItem.product_name}`);
  } else if (groceryItem.product_name) {
    parts.push(groceryItem.product_name);
  } else if (groceryItem.brand) {
    parts.push(groceryItem.brand);
  }
  
  // Variant (size, flavor, etc.)
  if (groceryItem.variant) {
    parts.push(groceryItem.variant);
  }
  
  // Category for context
  if (groceryItem.category) {
    parts.push(`(${groceryItem.category})`);
  }
  
  const productDescription = parts.join(' ') || 'grocery product';
  
  // Build additional context hints from ingredients/nutrition
  const contextHints: string[] = [];
  
  if (groceryItem.ingredients && groceryItem.ingredients.length > 0) {
    // Extract key visual characteristics from ingredients
    const ingredientsStr = groceryItem.ingredients.join(', ').toLowerCase();
    if (ingredientsStr.includes('chocolate')) contextHints.push('chocolate');
    if (ingredientsStr.includes('strawberry') || ingredientsStr.includes('berry')) contextHints.push('berry');
    if (ingredientsStr.includes('vanilla')) contextHints.push('vanilla');
    if (ingredientsStr.includes('mint')) contextHints.push('mint');
    if (ingredientsStr.includes('lemon') || ingredientsStr.includes('citrus')) contextHints.push('citrus');
    if (ingredientsStr.includes('apple')) contextHints.push('apple');
    if (ingredientsStr.includes('banana')) contextHints.push('banana');
    if (ingredientsStr.includes('orange')) contextHints.push('orange');
  }
  
  // Build prompt for minimal icon style
  let prompt = `A simple, minimal flat icon of ${productDescription}`;
  
  // Add visual context hints
  if (contextHints.length > 0) {
    prompt += ` with ${contextHints.join(' and ')} characteristics`;
  }
  
  prompt += `. 
  
Style: Clean vector-like icon, minimal detail, bold simple shapes, limited color palette.
Centered on a square canvas, high contrast, recognizable at small sizes.
Background: solid light neutral color (white or light gray).
NO TEXT. NO WORDS. NO LABELS. NO BRAND NAMES. NO BARCODES. NO WRITING OF ANY KIND.
Icon-only, not a photo.`;

  try {
    console.log('[generateIcon] Generating minimal icon for:', productDescription);
    if (contextHints.length > 0) {
      console.log('[generateIcon] Context hints:', contextHints.join(', '));
    }
    
    const response = await openai.images.generate({
      model: 'gpt-image-1',
      prompt: prompt,
      size: '1024x1024',
      quality: 'medium',
      output_format: 'jpeg',
      output_compression: 75,
      n: 1,
    });

    let imageBase64: string | undefined;
    
    if (response.data && Array.isArray(response.data) && response.data.length > 0) {
      imageBase64 = response.data[0]?.b64_json;
    }
    
    if (!imageBase64) {
      console.error('[generateIcon] No base64 image data returned. Response structure:', JSON.stringify({
        hasData: !!response.data,
        dataLength: Array.isArray(response.data) ? response.data.length : 'not array',
        dataType: typeof response.data,
        keys: Object.keys(response || {}),
        firstDataItem: Array.isArray(response.data) && response.data.length > 0 ? Object.keys(response.data[0] || {}) : 'no items'
      }, null, 2));
      return null;
    }

    console.log('[generateIcon] Icon generated, decoding base64 data...');
    
    const imageBuffer = Buffer.from(imageBase64, 'base64');
    console.log('[generateIcon] Optimized JPEG icon size:', imageBuffer.length, 'bytes');
    
    return imageBuffer;
  } catch (error) {
    console.error('[generateIcon] Error generating product icon:', error);
    return null;
  }
}
