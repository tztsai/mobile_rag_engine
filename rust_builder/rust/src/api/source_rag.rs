// Copyright 2025 mobile_rag_engine contributors
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License. You may obtain a copy of the License at
// https://opensource.org/licenses/MIT
//
// This software is provided "AS IS", without warranty of any kind, express or
// implied, including but not limited to the warranties of merchantability,
// fitness for a particular purpose, and noninfringement. In no event shall the
// authors or copyright holders be liable for any claim, damages, or other
// liability arising from the use of this software.
//
// CONTRIBUTOR GUIDELINES:
// This file is part of the core engine. Any modifications require owner approval.
// Please submit a PR with detailed explanation of changes before modifying.
//
//! Extended RAG API with sources and chunks for LLM-optimized context.

use crate::api::db_pool::get_connection;
use crate::api::hnsw_index::{build_hnsw_index, is_hnsw_index_loaded, search_hnsw};
use log::{debug, info};
use ndarray::Array1;
use rusqlite::params;
use sha2::{Digest, Sha256};

fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Initialize database with sources and chunks tables.
pub fn init_source_db() -> anyhow::Result<()> {
    info!("[init_source_db] Initializing database tables");
    let conn = get_connection()?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT UNIQUE,
                metadata TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                chunk_type TEXT DEFAULT 'general',
                embedding BLOB NOT NULL,
                FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
            )",
        [],
    )?;

    let has_chunk_type: bool = conn
        .prepare("SELECT chunk_type FROM chunks LIMIT 1")
        .is_ok();
    if !has_chunk_type {
        info!("[init_source_db] Migrating: adding chunk_type column");
        conn.execute(
            "ALTER TABLE chunks ADD COLUMN chunk_type TEXT DEFAULT 'general'",
            [],
        )?;
    }

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id)",
        [],
    )?;

    info!("[init_source_db] Tables created");
    Ok(())
}

#[derive(Debug, Clone)]
pub struct AddSourceResult {
    pub source_id: String,
    pub is_duplicate: bool,
    pub chunk_count: i32,
    pub message: String,
}

/// Add a source document with a specific ID (chunks added separately via add_chunks).
pub fn add_source(
    id: String,
    content: String,
    metadata: Option<String>,
) -> anyhow::Result<AddSourceResult> {
    info!("[add_source] Adding source {}, {} chars", id, content.len());

    let content_hash = hash_content(&content);
    let conn = get_connection()?;

    let existing: Option<String> = conn
        .query_row(
            "SELECT id FROM sources WHERE content_hash = ?1",
            params![content_hash],
            |row| row.get(0),
        )
        .ok();

    if let Some(existing_id) = existing {
        info!("[add_source] Duplicate found: {}", existing_id);
        return Ok(AddSourceResult {
            source_id: existing_id,
            is_duplicate: true,
            chunk_count: 0,
            message: format!("Source already exists (id={})", existing_id),
        });
    }

    conn.execute(
        "INSERT INTO sources (id, content, content_hash, metadata) VALUES (?1, ?2, ?3, ?4)",
        params![id, content, content_hash, metadata],
    )?;

    info!("[add_source] Created source: {}", id);

    Ok(AddSourceResult {
        source_id: id,
        is_duplicate: false,
        chunk_count: 0,
        message: "Source created".to_string(),
    })
}

#[derive(Debug, Clone)]
pub struct ChunkData {
    pub content: String,
    pub chunk_index: i32,
    pub start_pos: i32,
    pub end_pos: i32,
    pub chunk_type: String,
    pub embedding: Vec<f32>,
}

/// Add chunks for a source (uses transaction for atomicity).
pub fn add_chunks(source_id: &str, chunks: Vec<ChunkData>) -> anyhow::Result<i32> {
    info!(
        "[add_chunks] Adding {} chunks for source {}",
        chunks.len(),
        source_id
    );

    let mut conn = get_connection()?;
    let tx = conn.transaction()?;

    for chunk in &chunks {
        let mut embedding_bytes: Vec<u8> = Vec::with_capacity(chunk.embedding.len() * 4);
        for f in &chunk.embedding {
            embedding_bytes.extend_from_slice(&f.to_ne_bytes());
        }

        tx.execute(
            "INSERT INTO chunks (source_id, chunk_index, content, start_pos, end_pos, chunk_type, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![source_id, chunk.chunk_index, chunk.content, chunk.start_pos, chunk.end_pos, chunk.chunk_type, embedding_bytes],
        )?;
    }

    tx.commit()?;
    info!("[add_chunks] Added {} chunks", chunks.len());
    Ok(chunks.len() as i32)
}

/// Rebuild HNSW index from chunks table.
pub fn rebuild_chunk_hnsw_index() -> anyhow::Result<()> {
    info!("[rebuild_chunk_hnsw] Starting");
    let conn = get_connection()?;

    let mut stmt = conn.prepare("SELECT id, embedding FROM chunks")?;
    let points: Vec<(i64, Vec<f32>)> = stmt
        .query_map([], |row| {
            let id: i64 = row.get(0)?;
            let embedding_blob: Vec<u8> = row.get(1)?;
            let mut embedding = Vec::with_capacity(embedding_blob.len() / 4);
            for chunk in embedding_blob.chunks_exact(4) {
                embedding.push(f32::from_ne_bytes(chunk.try_into().unwrap()));
            }
            Ok((id, embedding))
        })?
        .filter_map(|r| r.ok())
        .collect();

    if !points.is_empty() {
        build_hnsw_index(points)?;
        // Note: save_hnsw_index needs db_path for marker file
        // This is acceptable as it's a one-time operation
        info!("[rebuild_chunk_hnsw] Built index");
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub struct ChunkSearchResult {
    pub chunk_id: i64,
    pub source_id: String,
    pub chunk_index: i32,
    pub content: String,
    pub chunk_type: String,
    pub similarity: f64,
    pub metadata: Option<String>,
}

/// Search chunks by embedding similarity.
pub fn search_chunks(
    query_embedding: Vec<f32>,
    top_k: u32,
) -> anyhow::Result<Vec<ChunkSearchResult>> {
    info!("[search_chunks] Searching, top_k={}", top_k);

    // HNSW index enabled - use O(log n) search when index is available
    // Falls back to linear scan if index not loaded

    let hnsw_loaded = is_hnsw_index_loaded();

    if !hnsw_loaded {
        // HNSW index is in-memory only - rebuild from DB on each app launch
        debug!("[search_chunks] HNSW not in memory, rebuilding from DB");
        rebuild_chunk_hnsw_index()?;
    }

    if !is_hnsw_index_loaded() {
        debug!("[search_chunks] Falling back to linear scan");
        return search_chunks_linear(query_embedding, top_k);
    }

    debug!("[search_chunks] Using HNSW index");

    let hnsw_results = search_hnsw(query_embedding, top_k as usize)?;
    let conn = get_connection()?;

    let mut results = Vec::new();
    for result in hnsw_results {
        let row: Option<(String, i32, String, String, Option<String>)> = conn
            .query_row(
                "SELECT c.source_id, c.chunk_index, c.content, COALESCE(c.chunk_type, 'general'), s.metadata 
                 FROM chunks c
                 LEFT JOIN sources s ON c.source_id = s.id
                 WHERE c.id = ?1",
                params![result.id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .ok();

        if let Some((source_id, chunk_index, content, chunk_type, metadata)) = row {
            results.push(ChunkSearchResult {
                chunk_id: result.id,
                source_id,
                chunk_index,
                content,
                chunk_type,
                similarity: 1.0 - result.distance as f64,
                metadata,
            });
        }
    }

    info!("[search_chunks] Found {} results", results.len());
    Ok(results)
}

fn search_chunks_linear(
    query_embedding: Vec<f32>,
    top_k: u32,
) -> anyhow::Result<Vec<ChunkSearchResult>> {
    let conn = get_connection()?;
    let mut stmt = conn.prepare(
        "SELECT c.id, c.source_id, c.chunk_index, c.content, COALESCE(c.chunk_type, 'general'), c.embedding, s.metadata 
         FROM chunks c
         LEFT JOIN sources s ON c.source_id = s.id"
    )?;
    
    let query_vec = Array1::from(query_embedding.clone());
    let query_norm = query_vec.mapv(|x| x * x).sum().sqrt();

    let mut candidates: Vec<(f64, i64, String, i32, String, String, Option<String>)> = Vec::new();

    let rows = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get::<_, Vec<u8>>(5)?, row.get(6)?))
    })?;

    for row in rows {
        let (id, source_id, chunk_index, content, chunk_type, embedding_blob, metadata): (
            i64, String, i32, String, String, Vec<u8>, Option<String>,
        ) = row?;

        let embedding: Vec<f32> = embedding_blob
            .chunks(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();

        if embedding.len() != query_embedding.len() {
            continue;
        }

        let target_vec = Array1::from(embedding);
        let target_norm = target_vec.mapv(|x| x * x).sum().sqrt();
        let dot_product = query_vec.dot(&target_vec);
        
        let similarity = if query_norm == 0.0 || target_norm == 0.0 { 0.0 }
        else { (dot_product / (query_norm * target_norm)) as f64 };
        
        candidates.push((similarity, id, source_id, chunk_index, content, chunk_type, metadata));
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    Ok(candidates.into_iter().take(top_k as usize)
        .map(|(sim, id, source_id, chunk_index, content, chunk_type, metadata)| ChunkSearchResult {
            chunk_id: id, source_id, chunk_index, content, chunk_type, similarity: sim, metadata,
        }).collect())
}

/// Get source document by ID.
pub fn get_source(source_id: &str) -> anyhow::Result<Option<String>> {
    let conn = get_connection()?;
    Ok(conn
        .query_row(
            "SELECT content FROM sources WHERE id = ?1",
            params![source_id],
            |row| row.get(0),
        )
        .ok())
}

/// Get all chunks for a source.
pub fn get_source_chunks(source_id: &str) -> anyhow::Result<Vec<String>> {
    let conn = get_connection()?;
    let mut stmt =
        conn.prepare("SELECT content FROM chunks WHERE source_id = ?1 ORDER BY chunk_index")?;
    let chunks: Vec<String> = stmt
        .query_map(params![source_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(chunks)
}

/// Get adjacent chunks by source_id and chunk_index range.
pub fn get_adjacent_chunks(
    source_id: &str,
    min_index: i32,
    max_index: i32,
) -> anyhow::Result<Vec<ChunkSearchResult>> {
    info!(
        "[get_adjacent_chunks] source={}, range={}..{}",
        source_id, min_index, max_index
    );
    let conn = get_connection()?;

    let mut stmt = conn.prepare(
        "SELECT c.id, c.source_id, c.chunk_index, c.content, COALESCE(c.chunk_type, 'general'), s.metadata 
         FROM chunks c 
         LEFT JOIN sources s ON c.source_id = s.id
         WHERE c.source_id = ?1 AND c.chunk_index >= ?2 AND c.chunk_index <= ?3 ORDER BY c.chunk_index"
    )?;

    let chunks: Vec<ChunkSearchResult> = stmt
        .query_map(params![source_id, min_index, max_index], |row| {
            Ok(ChunkSearchResult {
                chunk_id: row.get(0)?, source_id: row.get(1)?, chunk_index: row.get(2)?,
                content: row.get(3)?, chunk_type: row.get(4)?, similarity: 0.0,
                metadata: row.get(5)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    info!("[get_adjacent_chunks] Found {} chunks", chunks.len());
    Ok(chunks)
}

/// Delete a source and all its chunks.
pub fn delete_source(source_id: &str) -> anyhow::Result<()> {
    let conn = get_connection()?;
    conn.execute(
        "DELETE FROM chunks WHERE source_id = ?1",
        params![source_id],
    )?;
    conn.execute("DELETE FROM sources WHERE id = ?1", params![source_id])?;
    info!("[delete_source] Deleted source {}", source_id);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SourceStats {
    pub source_count: i64,
    pub chunk_count: i64,
}

pub fn get_source_stats() -> anyhow::Result<SourceStats> {
    let conn = get_connection()?;
    let source_count: i64 = conn.query_row("SELECT COUNT(*) FROM sources", [], |row| row.get(0))?;
    let chunk_count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    Ok(SourceStats {
        source_count,
        chunk_count,
    })
}

#[derive(Debug, Clone)]
pub struct ChunkForReembedding {
    pub chunk_id: i64,
    pub content: String,
}

/// Get all chunk IDs and contents for re-embedding.
pub fn get_all_chunk_ids_and_contents() -> anyhow::Result<Vec<ChunkForReembedding>> {
    info!("[get_all_chunk_ids_and_contents] Starting");
    let conn = get_connection()?;
    let mut stmt = conn.prepare("SELECT id, content FROM chunks ORDER BY id")?;
    let chunks: Vec<ChunkForReembedding> = stmt
        .query_map([], |row| {
            Ok(ChunkForReembedding {
                chunk_id: row.get(0)?,
                content: row.get(1)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
    info!(
        "[get_all_chunk_ids_and_contents] Found {} chunks",
        chunks.len()
    );
    Ok(chunks)
}

/// Update embedding for a single chunk.
pub fn update_chunk_embedding(chunk_id: i64, embedding: Vec<f32>) -> anyhow::Result<()> {
    let conn = get_connection()?;
    let mut embedding_bytes: Vec<u8> = Vec::with_capacity(embedding.len() * 4);
    for f in &embedding {
        embedding_bytes.extend_from_slice(&f.to_ne_bytes());
    }
    conn.execute(
        "UPDATE chunks SET embedding = ?1 WHERE id = ?2",
        params![embedding_bytes, chunk_id],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::db_pool::{init_db_pool, close_db_pool};
    use crate::api::hnsw_index::clear_hnsw_index;

    #[test]
    fn test_metadata_retrieval() {
        // 1. Setup
        let db_path = std::env::temp_dir().join("test_metadata.db");
        let _ = std::fs::remove_file(&db_path);
        
        init_db_pool(db_path.to_str().unwrap().to_string(), 1).unwrap();
        init_source_db().unwrap();
        clear_hnsw_index();

        // 2. Add Source with Metadata
        let metadata = r#"{"author": "Test Author", "year": 2025}"#;
        let source_res = add_source("Test Content".to_string(), Some(metadata.to_string())).unwrap();
        
        let chunk = ChunkData {
            content: "Test Chunk".to_string(),
            chunk_index: 0,
            start_pos: 0,
            end_pos: 10,
            chunk_type: "text".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0], // 4 dims
        };
        add_chunks(source_res.source_id, vec![chunk]).unwrap();

        // 3. Search (Linear Scan)
        let results = search_chunks(vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        // 4. Verify Metadata
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata, Some(metadata.to_string()));
        assert_eq!(results[0].source_id, source_res.source_id);

        // 5. Cleanup
        close_db_pool();
        let _ = std::fs::remove_file(db_path);
    }
}
