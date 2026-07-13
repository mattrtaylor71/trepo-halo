export interface FeedEventRecord {
    owner: string;
    device_id: string;
    user_id: string;
    job_id: string;
    event_type: string;
    action?: string | null;
    title?: string | null;
    brand?: string | null;
    image_url?: string | null;
    product_image_url?: string | null;
    source_table?: string | null;
    metadata?: Record<string, unknown> | null;
}
export declare function writeFeedEvent(record: FeedEventRecord): Promise<void>;
