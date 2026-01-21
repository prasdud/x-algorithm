# X For You Feed Algorithm - Complete Project Documentation

This document provides comprehensive context for the entire codebase, covering all components, their interactions, and implementation details. It serves as the primary reference for both humans and AI agents working on this project.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Component Deep Dives](#component-deep-dives)
4. [Data Models](#data-models)
5. [Pipeline Execution Flow](#pipeline-execution-flow)
6. [External Dependencies](#external-dependencies)
7. [Configuration & Constants](#configuration--constants)
8. [Development Guidelines](#development-guidelines)

---

## Project Overview

The **X For You Feed Algorithm** is a production-grade recommendation system that powers the "For You" timeline on X (formerly Twitter). It combines:

- **In-Network Content**: Posts from accounts the user follows (served by Thunder)
- **Out-of-Network Content**: ML-discovered content from the global corpus (served by Phoenix)

The system processes user requests through a multi-stage pipeline that retrieves, filters, hydrates, scores, and ranks content using a Grok-based transformer model.

### Key Design Philosophy

**No Hand-Engineered Features**: The system relies entirely on the Grok-based transformer to learn relevance from user engagement sequences. Traditional recommendation systems require extensive feature engineering (recency scores, author authority, content heuristics). This system replaces all of that with pure deep learning.

### Technology Stack

| Component | Language | Framework |
|-----------|----------|-----------|
| Home Mixer | Rust | Tokio, tonic (gRPC) |
| Thunder | Rust | Tokio, tonic, Kafka |
| Phoenix | Python | JAX, Haiku |
| Candidate Pipeline | Rust | Trait-based framework |

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    CLIENT REQUEST                                             │
│                                (user opens For You feed)                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      HOME MIXER (Rust)                                        │
│                              gRPC: ScoredPostsService/get_scored_posts                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              CANDIDATE PIPELINE                                           │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │  │
│  │  │   QUERY      │───▶│   SOURCES    │───▶│  HYDRATORS   │───▶│   FILTERS    │          │  │
│  │  │  HYDRATORS   │    │              │    │              │    │              │          │  │
│  │  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘          │  │
│  │         │                   │                    │                   │                   │  │
│  │         ▼                   ▼                    ▼                   ▼                   │  │
│  │  • UserActionSeq     • ThunderSource      • CoreData          • DropDuplicates         │  │
│  │    QueryHydrator        (In-Network)        CandidateHydrator    • AgeFilter             │  │
│  │  • UserFeatures      • PhoenixSource      • Gizmoduck          • SelfTweetFilter        │  │
│  │    QueryHydrator        (Out-of-Network)   CandidateHydrator    • MutedKeywordFilter     │  │
│  │                                            • VFCandidate       • AuthorSocialgraph      │  │
│  │                                            Hydrator             Filter                  │  │
│  │                                                                                          │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                               │  │
│  │  │   SCORERS    │───▶│  SELECTOR    │───▶│ POST-SELECT │                               │  │
│  │  │              │    │              │    │    STAGE     │                               │  │
│  │  └──────────────┘    └──────────────┘    └──────────────┘                               │  │
│  │         │                   │                    │                                      │  │
│  │         ▼                   ▼                    ▼                                      │  │
│  │  • PhoenixScorer      • TopKScore         • VFCandidateHydrator                          │  │
│  │    (Grok ML)          Selector            • VFFilter                                     │  │
│  │  • WeightedScorer                           • DedupConversationFilter                     │  │
│  │    (Combine probs)                                                                       │  │
│  │  • AuthorDiversity      ┌────────────────────────────────────────┐                      │  │
│  │    Scorer               │        SIDE EFFECTS (async)            │                      │  │
│  │  • OONScorer            • CacheRequestInfoSideEffect             │                      │  │
│  │                          └────────────────────────────────────────┘                      │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐        ┌──────────────────────────────────────────────────────────┐
│   THUNDER (Rust)     │        │         PHOENIX (Python/JAX)                               │
│   In-Network Posts   │        │         ML-Based Content Discovery                         │
├──────────────────────┤        ├──────────────────────────────────────────────────────────┤
│ • PostStore          │        │                                                          │
│   - In-memory        │        │  ┌────────────────────────────────────────────────┐      │
│     storage          │        │  │        RETRIEVAL (Two-Tower)                    │      │
│ • Kafka Ingestion    │        │  │  ┌──────────┐      ┌──────────────┐            │      │
│   - TweetCreateEvent │        │  │  │   User   │      │   Candidate  │            │      │
│   - TweetDeleteEvent │        │  │  │   Tower  │      │    Tower     │            │      │
│ • gRPC API           │        │  │  │ (Embed   │      │  (Project &  │            │      │
│   - GetInNetworkPosts│        │  │  │  history)│      │   normalize) │            │      │
│                      │        │  │  └──────────┘      └──────────────┘            │      │
│                      │        │  │         │                 │                    │      │
│                      │        │  │         └────────┬────────┘                    │      │
│                      │        │  │                  ▼                            │      │
│                      │        │  │         ANN Search (dot product)               │      │
│                      │        │  └────────────────────────────────────────────────┘      │
│                      │        │                                                          │
│                      │        │  ┌────────────────────────────────────────────────┐      │
│                      │        │  │        RANKING (Transformer)                   │      │
│                      │        │  │                                                   │      │
│                      │        │  │   User + History + Candidates                    │      │
│                      │        │  │          │                                        │      │
│                      │        │  │          ▼                                        │      │
│                      │        │  │   Grok Transformer (with candidate isolation)     │      │
│                      │        │  │          │                                        │      │
│                      │        │  │          ▼                                        │      │
│                      │        │  │   Action Probabilities (like, reply, ...)        │      │
│                      │        │  └────────────────────────────────────────────────┘      │
│                      │        └──────────────────────────────────────────────────────────┘
└──────────────────────┘
```

---

## Component Deep Dives

### 1. Candidate Pipeline Framework

**Location**: `candidate-pipeline/`

A reusable, trait-based framework for building recommendation pipelines in Rust.

#### File Structure

```
candidate-pipeline/
├── lib.rs                      # Module exports
├── candidate_pipeline.rs       # Main pipeline orchestration
├── source.rs                   # Source trait definition
├── hydrator.rs                 # Hydrator trait definition
├── query_hydrator.rs           # QueryHydrator trait definition
├── filter.rs                   # Filter trait definition
├── scorer.rs                   # Scorer trait definition
├── selector.rs                 # Selector trait definition
├── side_effect.rs              # SideEffect trait definition
└── util/                       # Utility functions
```

#### Core Trait Definitions

**Source** (`source.rs`):
```rust
pub trait Source<Q, C>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, query: &Q) -> bool;
    async fn get_candidates(&self, query: &Q) -> Result<Vec<C>, String>;
    fn name(&self) -> &'static str;
}
```
- **Purpose**: Fetch candidates from a data source
- **Behavior**: All sources run in parallel, results are concatenated
- **Examples**: `ThunderSource`, `PhoenixSource`

**Hydrator** (`hydrator.rs`):
```rust
pub trait Hydrator<Q, C>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, query: &Q) -> bool;
    async fn hydrate(&self, query: &Q, candidates: &[C]) -> Result<Vec<C>, String>;
    fn update(&self, candidate: &mut C, hydrated: C);
    fn update_all(&self, candidates: &mut [C], hydrated: Vec<C>);
    fn name(&self) -> &'static str;
}
```
- **Purpose**: Enrich candidates with additional data
- **Behavior**: Run in parallel, must preserve order and length
- **Critical Constraint**: Cannot drop candidates (use filters for that)
- **Examples**: `CoreDataCandidateHydrator`, `GizmoduckCandidateHydrator`

**Filter** (`filter.rs`):
```rust
pub trait Filter<Q, C>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, query: &Q) -> bool;
    async fn filter(&self, query: &Q, candidates: Vec<C>) -> Result<FilterResult<C>, String>;
    fn name(&self) -> &'static str;
}

pub struct FilterResult<C> {
    pub kept: Vec<C>,
    pub removed: Vec<C>,
}
```
- **Purpose**: Remove candidates that shouldn't be shown
- **Behavior**: Run sequentially, each filter sees the output of the previous
- **Examples**: `AgeFilter`, `VFFilter`, `DropDuplicatesFilter`

**Scorer** (`scorer.rs`):
```rust
pub trait Scorer<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, query: &Q) -> bool;
    async fn score(&self, query: &Q, candidates: &[C]) -> Result<Vec<C>, String>;
    fn update(&self, candidate: &mut C, scored: C);
    fn update_all(&self, candidates: &mut [C], scored: Vec<C>);
    fn name(&self) -> &'static str;
}
```
- **Purpose**: Compute scores for ranking
- **Behavior**: Run sequentially, each scorer can modify scores
- **Examples**: `PhoenixScorer`, `WeightedScorer`, `AuthorDiversityScorer`

**Selector** (`selector.rs`):
```rust
pub trait Selector<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, query: &Q) -> bool;
    fn select(&self, query: &Q, candidates: Vec<C>) -> Vec<C>;
    fn score(&self, candidate: &C) -> f64;
    fn sort(&self, candidates: Vec<C>) -> Vec<C>;
    fn size(&self) -> Option<usize>;
    fn name(&self) -> &'static str;
}
```
- **Purpose**: Sort and truncate to top K
- **Behavior**: Single selector runs after all scorers
- **Example**: `TopKScoreSelector`

**SideEffect** (`side_effect.rs`):
```rust
pub trait SideEffect<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, query: Arc<Q>) -> bool;
    async fn run(&self, input: Arc<SideEffectInput<Q, C>>) -> Result<(), String>;
    fn name(&self) -> &'static str;
}

pub struct SideEffectInput<Q, C> {
    pub query: Arc<Q>,
    pub selected_candidates: Vec<C>,
}
```
- **Purpose**: Async operations that don't affect the returned result
- **Behavior**: Run in parallel in background after response is returned
- **Example**: `CacheRequestInfoSideEffect`

#### Pipeline Execution Sequence

The `CandidatePipeline::execute()` method (`candidate_pipeline.rs:53-92`) orchestrates the entire flow:

```
1. hydrate_query()       → Run QueryHydrators in parallel, merge results
2. fetch_candidates()    → Run Sources in parallel, collect all candidates
3. hydrate()             → Run Hydrators in parallel, enrich candidates
4. filter()              → Run Filters sequentially
5. score()               → Run Scorers sequentially
6. select()              → Sort by score, truncate to top K
7. hydrate_post_selection() → Run post-selection Hydrators in parallel
8. filter_post_selection() → Run post-selection Filters
9. run_side_effects()    → Spawn background tasks
10. Return PipelineResult
```

---

### 2. Home Mixer

**Location**: `home-mixer/`

The main orchestration service that receives user requests and executes the recommendation pipeline.

#### File Structure

```
home-mixer/
├── main.rs                          # gRPC server entry point
├── lib.rs                           # Module exports
├── server.rs                        # gRPC service implementation
├── candidate_pipeline/              # Pipeline-specific code
│   ├── phoenix_candidate_pipeline.rs  # Concrete pipeline implementation
│   ├── query.rs                      # Query type definition
│   ├── candidate.rs                  # Candidate type definition
│   ├── candidate_features.rs         # Candidate feature structures
│   └── query_features.rs             # Query feature structures
├── query_hydrators/                 # Query enrichment components
│   ├── user_action_seq_query_hydrator.rs
│   └── user_features_query_hydrator.rs
├── sources/                         # Candidate sources
│   ├── thunder_source.rs
│   └── phoenix_source.rs
├── candidate_hydrators/             # Candidate enrichment
│   ├── core_data_candidate_hydrator.rs
│   ├── gizmoduck_hydrator.rs
│   ├── video_duration_candidate_hydrator.rs
│   ├── subscription_hydrator.rs
│   ├── in_network_candidate_hydrator.rs
│   └── vf_candidate_hydrator.rs
├── filters/                         # Content filtering
│   ├── drop_duplicates_filter.rs
│   ├── core_data_hydration_filter.rs
│   ├── age_filter.rs
│   ├── self_tweet_filter.rs
│   ├── retweet_deduplication_filter.rs
│   ├── ineligible_subscription_filter.rs
│   ├── previously_seen_posts_filter.rs
│   ├── previously_served_posts_filter.rs
│   ├── muted_keyword_filter.rs
│   ├── author_socialgraph_filter.rs
│   ├── vf_filter.rs
│   └── dedup_conversation_filter.rs
├── scorers/                         # Ranking components
│   ├── phoenix_scorer.rs
│   ├── weighted_scorer.rs
│   ├── author_diversity_scorer.rs
│   └── oon_scorer.rs
├── selectors/                       # Candidate selection
│   └── top_k_score_selector.rs
├── side_effects/                    # Async post-processing
│   └── cache_request_info_side_effect.rs
├── clients/                         # External service clients (excluded from OSS)
└── params.rs                        # Configuration constants (excluded from OSS)
```

#### Key Data Structures

**ScoredPostsQuery** (`candidate_pipeline/query.rs`):
```rust
pub struct ScoredPostsQuery {
    pub user_id: i64,
    pub client_app_id: i32,
    pub country_code: String,
    pub language_code: String,
    pub seen_ids: Vec<i64>,              // Posts user has seen
    pub served_ids: Vec<i64>,            // Posts already served this session
    pub in_network_only: bool,           // Skip Phoenix retrieval
    pub is_bottom_request: bool,         // Is this a pagination request
    pub bloom_filter_entries: Vec<ImpressionBloomFilterEntry>,
    pub user_action_sequence: Option<UserActionSequence>,  // Engagement history
    pub user_features: UserFeatures,     // Following list, preferences
    pub request_id: String,              // For tracing/logging
}
```

**PostCandidate** (`candidate_pipeline/candidate.rs`):
```rust
pub struct PostCandidate {
    // Core identification
    pub tweet_id: i64,
    pub author_id: u64,
    pub tweet_text: String,

    // Thread information
    pub in_reply_to_tweet_id: Option<u64>,
    pub retweeted_tweet_id: Option<u64>,
    pub retweeted_user_id: Option<u64>,
    pub ancestors: Vec<u64>,

    // Scoring fields
    pub phoenix_scores: PhoenixScores,
    pub weighted_score: Option<f64>,
    pub score: Option<f64>,              // Final score after all scorers

    // Metadata
    pub served_type: Option<ServedType>, // Source type
    pub in_network: Option<bool>,
    pub prediction_request_id: Option<u64>,
    pub last_scored_at_ms: Option<u64>,

    // Hydrated fields
    pub video_duration_ms: Option<i32>,
    pub author_followers_count: Option<i32>,
    pub author_screen_name: Option<String>,
    pub retweeted_screen_name: Option<String>,
    pub visibility_reason: Option<FilteredReason>,
    pub subscription_author_id: Option<u64>,
}
```

**PhoenixScores** (`candidate_pipeline/candidate.rs:29-51`):
```rust
pub struct PhoenixScores {
    // Engagement actions (positive)
    pub favorite_score: Option<f64>,
    pub reply_score: Option<f64>,
    pub retweet_score: Option<f64>,
    pub quote_score: Option<f64>,
    pub click_score: Option<f64>,
    pub profile_click_score: Option<f64>,
    pub vqv_score: Option<f64>,          // Video quality view
    pub photo_expand_score: Option<f64>,
    pub share_score: Option<f64>,
    pub share_via_dm_score: Option<f64>,
    pub share_via_copy_link_score: Option<f64>,
    pub dwell_score: Option<f64>,
    pub follow_author_score: Option<f64>,

    // Negative feedback
    pub not_interested_score: Option<f64>,
    pub block_author_score: Option<f64>,
    pub mute_author_score: Option<f64>,
    pub report_score: Option<f64>,

    // Continuous values
    pub dwell_time: Option<f64>,
}
```

#### Pipeline Construction

The `PhoenixCandidatePipeline::build_with_clients()` method (`candidate_pipeline/phoenix_candidate_pipeline.rs:73-160`) constructs the complete pipeline:

**Query Hydrators**:
- `UserActionSeqQueryHydrator`: Fetches user's engagement history (likes, replies, retweets)
- `UserFeaturesQueryHydrator`: Fetches following list and user preferences

**Sources**:
- `ThunderSource`: Fetches posts from followed accounts
- `PhoenixSource`: Fetches ML-discovered posts from global corpus

**Hydrators**:
- `InNetworkCandidateHydrator`: Marks candidates as in-network/out-of-network
- `CoreDataCandidateHydrator`: Fetches tweet text, media, creation time
- `VideoDurationCandidateHydrator`: Fetches video duration for video posts
- `SubscriptionHydrator`: Checks paywall/subscriber status
- `GizmoduckCandidateHydrator`: Fetches author info (username, verification, follower count)

**Pre-Scoring Filters** (run in order):
1. `DropDuplicatesFilter`: Removes duplicate tweet IDs
2. `CoreDataHydrationFilter`: Drops candidates that failed hydration
3. `AgeFilter`: Removes posts older than `MAX_POST_AGE`
4. `SelfTweetFilter`: Removes user's own posts
5. `RetweetDeduplicationFilter`: Removes duplicate retweets of same content
6. `IneligibleSubscriptionFilter`: Removes paywalled content user can't access
7. `PreviouslySeenPostsFilter`: Removes posts user has recently seen
8. `PreviouslyServedPostsFilter`: Removes posts already served in session
9. `MutedKeywordFilter`: Filters posts containing muted keywords
10. `AuthorSocialgraphFilter`: Removes posts from blocked/muted authors

**Scorers** (run in order):
1. `PhoenixScorer`: Calls ML model to get action probabilities
2. `WeightedScorer`: Combines probabilities into weighted score
3. `AuthorDiversityScorer`: Attenuates repeated author scores for diversity
4. `OONScorer`: Adjusts scores for out-of-network content

**Selector**:
- `TopKScoreSelector`: Sorts by final score, returns top `RESULT_SIZE` candidates

**Post-Selection Hydrators**:
- `VFCandidateHydrator`: Fetches visibility filtering results

**Post-Selection Filters**:
- `VFFilter`: Removes posts that fail visibility filtering (spam, abuse, deleted)
- `DedupConversationFilter`: Deduplicates conversation threads

**Side Effects**:
- `CacheRequestInfoSideEffect`: Caches request data for analytics/debugging

#### Scorer Implementations

**PhoenixScorer** (`scorers/phoenix_scorer.rs`):
- Calls Phoenix gRPC service with user action sequence and candidates
- Returns probability predictions for all engagement types
- Uses retweet ID for retweets so predictions apply to original content

**WeightedScorer** (`scorers/weighted_scorer.rs`):
- Combines all Phoenix scores into a single relevance score
- Formula: `weighted_score = Σ (weight_i × P(action_i))`
- Applies special handling for video quality view (only if video duration > threshold)
- Offsets negative scores to prevent negative content from dominating

**AuthorDiversityScorer** (`scorers/author_diversity_scorer.rs`):
- Ensures feed diversity by attenuating repeated author scores
- Formula: `adjusted_score = base_score × (floor + (1-floor) × decay^position)`
- Position increments each time the same author appears in sorted candidates

**OONScorer** (`scorers/oon_scorer.rs`):
- Applies final adjustment for out-of-network content
- May boost or suppress OON content based on user preferences

---

### 3. Thunder

**Location**: `thunder/`

An in-memory post store that tracks recent posts from all users for sub-millisecond retrieval.

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        THUNDER                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌────────────────┐         ┌─────────────────┐           │
│   │   KAFKA        │         │   POSTSTORE     │           │
│   │   INGESTION    │────────▶│   (in-memory)   │           │
│   │                │         │                 │           │
│   │ • CreateEvent  │         │ ┌─────────────┐ │           │
│   │ • DeleteEvent  │         │ │  posts      │ │  HashMap  │
│   └────────────────┘         │ │  (by ID)    │ │  i64→Post │
│                              │ └─────────────┘ │           │
│                              │                 │           │
│                              │ ┌─────────────┐ │           │
│                              │ │original_    │ │           │
│                              │ │posts_by_user│ │  HashMap  │
│                              │ └─────────────┘ │  i64→Deque │
│                              │                 │           │
│                              │ ┌─────────────┐ │           │
│                              │ │secondary_   │ │           │
│                              │ │posts_by_user│ │  HashMap  │
│                              │ └─────────────┘ │  i64→Deque │
│                              │                 │           │
│                              │ ┌─────────────┐ │           │
│                              │ │video_       │ │           │
│                              │ │posts_by_user│ │  HashMap  │
│                              │ └─────────────┘ │  i64→Deque │
│                              └─────────────────┘           │
│                                                             │
│   ┌────────────────┐         ┌─────────────────┐           │
│   │   gRPC API     │◀────────│   STRATOCLIENT  │           │
│   │                │         │   (following)   │           │
│   │ GetInNetwork   │         └─────────────────┘           │
│   │ Posts          │                                      │
│   └────────────────┘                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### File Structure

```
thunder/
├── main.rs                          # Service entry point
├── lib.rs                           # Module exports
├── thunder_service.rs               # gRPC service implementation
├── posts/
│   └── post_store.rs                # Core in-memory storage
├── kafka/
│   ├── mod.rs                       # Kafka module exports
│   ├── kafka_utils.rs               # Kafka consumer utilities
│   ├── tweet_events_listener.rs     # V1 event processing
│   ├── tweet_events_listener_v2.rs  # V2 event processing (current)
│   └── utils.rs                     # Helper functions
├── config/                          # Configuration constants
│   ├── mod.rs
│   └── constants.rs
├── strato_client.rs                 # Database client for following lists
└── deserializer.rs                  # Kafka message deserialization
```

#### PostStore Core

The `PostStore` (`posts/post_store.rs`) maintains three timelines per user:

1. **original_posts_by_user**: Non-reply, non-retweet posts
2. **secondary_posts_by_user**: Replies and retweets
3. **video_posts_by_user**: Video posts (subset of original)

**Key Operations**:

- `insert_posts()`: Add new posts to store
- `mark_as_deleted()`: Record post deletions
- `get_all_posts_by_users()`: Retrieve posts for multiple users
- `get_videos_by_users()`: Retrieve only video posts
- `trim_old_posts()`: Remove posts older than retention period
- `start_auto_trim()`: Background task to periodically clean up

**TinyPost Optimization**:
```rust
pub struct TinyPost {
    pub post_id: i64,
    pub created_at: i64,
}
```
Instead of storing full post objects in timelines, Thunder stores only ID + timestamp. Full `LightPost` data is looked up from the main `posts` map when serving requests.

**Retention Strategy**:
- Posts older than retention period are automatically removed
- Configurable via `post_retention_seconds` CLI argument
- Default: 2 days (172,800 seconds)

#### Kafka Ingestion

Thunder consumes tweet events from Kafka:

**TweetCreateEvent**: New post created
- Adds to `posts` map
- Adds to appropriate user timeline based on type
- For retweets: if source post has video, mark as video

**TweetDeleteEvent**: Post deleted
- Removes from `posts` map
- Marks as deleted in `deleted_posts` set
- Timeline entries are cleaned during auto-trim

**Multi-threaded Processing**:
- Configurable number of consumer threads
- Each thread handles a subset of Kafka partitions
- Semaphore limits concurrent processing to prevent overload

#### gRPC API

**GetInNetworkPosts** Request:
```protobuf
message GetInNetworkPostsRequest {
    uint64 user_id = 1;
    repeated uint64 following_user_ids = 2;
    uint32 max_results = 3;
    repeated uint64 exclude_tweet_ids = 4;
    string algorithm = 5;
    bool debug = 6;
    bool is_video_request = 7;
}
```

**Response**:
```protobuf
message GetInNetworkPostsResponse {
    repeated LightPost posts = 1;
}

message LightPost {
    int64 post_id = 1;
    uint64 author_id = 2;
    int64 created_at = 3;
    int64 in_reply_to_post_id = 4;
    int64 in_reply_to_user_id = 5;
    bool is_retweet = 6;
    bool is_reply = 7;
    int64 source_post_id = 8;
    int64 source_user_id = 9;
    bool has_video = 10;
    int64 conversation_id = 11;
}
```

**Scoring**: Posts are sorted by recency (newest first). No ML scoring is done in Thunder.

**Rate Limiting**: Uses a semaphore to reject requests when at capacity, returning `RESOURCE_EXHAUSTED` status immediately.

---

### 4. Phoenix

**Location**: `phoenix/`

ML components for retrieval (finding candidates) and ranking (scoring candidates).

#### Architecture Overview

Phoenix uses a two-stage approach:

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHOENIX PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌────────────────────────────────────────────────────────┐   │
│   │                  STAGE 1: RETRIEVAL                    │   │
│   │                    (Two-Tower)                         │   │
│   │                                                         │   │
│   │   User Tower            Candidate Tower                │   │
│   │  (Embed history)      (Project & normalize)            │   │
│   │        │                       │                       │   │
│   │        ▼                       ▼                       │   │
│   │   User Embedding      Corpus Embeddings                │   │
│   │   [B, D]              [N, D]                           │   │
│   │        │                       │                       │   │
│   │        └───────────┬───────────┘                       │   │
│   │                    ▼                                   │   │
│   │         Dot Product Similarity                         │   │
│   │                    │                                   │   │
│   │                    ▼                                   │   │
│   │            Top-K Selection                            │   │
│   │              [B, K] candidates                         │   │
│   └────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   ┌────────────────────────────────────────────────────────┐   │
│   │                   STAGE 2: RANKING                     │   │
│   │              (Transformer with Isolation)               │   │
│   │                                                         │   │
│   │   User + History + K Candidates                         │   │
│   │        │                                                │   │
│   │        ▼                                                │   │
│   │   Grok Transformer                                     │   │
│   │   (candidates cannot attend to each other)             │   │
│   │        │                                                │   │
│   │        ▼                                                │   │
│   │   Action Probabilities [B, K, A]                        │   │
│   │                                                         │   │
│   └────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### File Structure

```
phoenix/
├── README.md                        # Detailed ML architecture docs
├── pyproject.toml                   # Python project config
├── recsys_model.py                  # Ranking transformer model
├── recsys_retrieval_model.py        # Retrieval two-tower model
├── grok.py                          # Grok-1 transformer implementation
├── runners.py                       # Model inference utilities
├── run_ranker.py                    # Ranking entry point
├── run_retrieval.py                 # Retrieval entry point
├── test_recsys_model.py             # Unit tests
└── test_recsys_retrieval_model.py   # Unit tests
```

#### Hash-Based Embeddings

Both models use hash-based embeddings for efficient lookup:

**HashConfig** (`recsys_model.py:32-38`):
```python
@dataclass
class HashConfig:
    num_user_hashes: int = 2
    num_item_hashes: int = 2
    num_author_hashes: int = 2
```

Multiple hash functions map sparse IDs to dense embeddings. This allows:
- Fixed-size embedding tables regardless of vocabulary size
- O(1) lookup for any ID (even unseen ones)
- Memory-efficient storage

**Embedding Combination**:
- User: `num_user_hashes × D` → projected to `D`
- Items (posts): `num_item_hashes × D` → projected to `D`
- Authors: `num_author_hashes × D` → projected to `D`

#### Ranking Model (`recsys_model.py`)

**PhoenixModel** - A transformer model that predicts engagement probabilities.

**Input Construction** (`build_inputs()`):
1. User embeddings are built from `user_hashes`
2. History embeddings combine: `post_hashes + author_hashes + actions + product_surface`
3. Candidate embeddings combine: `post_hashes + author_hashes + product_surface`
4. All concatenated: `[user, history, candidates]`

**Transformer Forward Pass**:
- Uses candidate isolation: candidates attend to user+history but not each other
- Output embeddings are layer-normalized
- Candidate embeddings are extracted and unembedded to action logits

**Candidate Isolation Mask** (`grok.py:39-71`):
```python
def make_recsys_attn_mask(seq_len, candidate_start_offset, dtype):
    # Start with causal mask
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))

    # Zero out candidate-to-candidate attention
    attn_mask = causal_mask
    attn_mask = attn_mask.at[candidate_offset:, candidate_offset:].set(0)

    # Add back self-attention for candidates
    candidate_indices = jnp.arange(candidate_start_offset, seq_len)
    attn_mask = attn_mask.at[candidate_indices, candidate_indices].set(1)

    return attn_mask
```

This ensures score consistency: a post's score doesn't depend on what other posts are in the batch.

#### Retrieval Model (`recsys_retrieval_model.py`)

**PhoenixRetrievalModel** - Two-tower architecture for ANN search.

**User Tower**:
- Uses same transformer as ranking (without candidates)
- Encodes user + history into single embedding
- Output is L2-normalized

**Candidate Tower** (`CandidateTower`):
- Two-layer MLP: `post_hash + author_hash` → hidden → normalized
- Uses SiLU activation
- Output is L2-normalized

**Retrieval** (`_retrieve_top_k()`):
- Dot product between user embedding and all corpus embeddings
- `top_k()` operation returns highest similarity candidates

#### Grok Transformer (`grok.py`)

Adapted from Grok-1 open source release.

**Key Components**:
- `MultiHeadAttention`: Grouped-query attention (GQA)
- `RotaryEmbedding`: RoPE positional encoding
- `RMSNorm`: Root Mean Square Layer Normalization
- `DenseBlock`: SwiGLU-style feed-forward (two projections, gating)

**Configuration**:
```python
@dataclass
class TransformerConfig:
    emb_size: int
    key_size: int
    num_q_heads: int
    num_kv_heads: int
    num_layers: int
    widening_factor: float = 4.0
    attn_output_multiplier: float = 1.0
```

**Grouped-Query Attention**: Multiple query heads share key/value heads, reducing memory while maintaining expressiveness.

#### Action Types Predicted

The model predicts probabilities for:

| Action | Type | Description |
|--------|------|-------------|
| ServerTweetFav | Positive | Like/favorite |
| ServerTweetReply | Positive | Reply |
| ServerTweetRetweet | Positive | Retweet |
| ServerTweetQuote | Positive | Quote tweet |
| ClientTweetClick | Positive | Click on tweet |
| ClientTweetClickProfile | Positive | Click on author profile |
| ClientTweetVideoQualityView | Positive | Video quality view (3+ seconds) |
| ClientTweetPhotoExpand | Positive | Expand photo |
| ClientTweetShare | Positive | Share tweet |
| ClientTweetClickSendViaDirectMessage | Positive | Share via DM |
| ClientTweetShareViaCopyLink | Positive | Share via copy link |
| ClientTweetRecapDwelled | Positive | Dwell on recap |
| ClientTweetFollowAuthor | Positive | Follow author |
| ClientTweetNotInterestedIn | Negative | Not interested |
| ClientTweetBlockAuthor | Negative | Block author |
| ClientTweetMuteAuthor | Negative | Mute author |
| ClientTweetReport | Negative | Report tweet |
| DwellTime | Continuous | Time spent viewing |

---

## Data Models

### Request Flow

1. **Client Request** → Home Mixer gRPC
   - `user_id`, `seen_ids`, `served_ids`, `country_code`, etc.

2. **Query Hydration**
   - Fetch `UserActionSequence`: Recent engagements (tweet_id, author_id, action_type)
   - Fetch `UserFeatures`: Following list, preferences

3. **Candidate Sourcing**
   - Thunder: Posts from followed accounts
   - Phoenix: ML-discovered posts

4. **Candidate Hydration**
   - Core metadata (text, media, timestamps)
   - Author info (username, verification)
   - Video duration
   - Subscription status
   - Network context (in-network vs out-of-network)

5. **Filtering** (Pre-scoring)
   - Remove duplicates, old posts, self-posts
   - Filter blocked/muted, muted keywords
   - Remove previously seen/served

6. **Scoring**
   - Phoenix ML predictions
   - Weighted combination
   - Author diversity adjustment
   - OON adjustment

7. **Selection**
   - Sort by score
   - Return top K (default ~100)

8. **Post-Selection**
   - Visibility filtering
   - Conversation deduplication

9. **Response**
   - List of scored posts with metadata

---

## Pipeline Execution Flow

### Complete Request Timeline

```
T+0ms:    Request received at Home Mixer
T+5ms:    Query hydrators start (parallel)
T+20ms:   User action sequence fetched
T+25ms:   User features fetched (following list)
T+30ms:   Sources start (parallel)
T+50ms:   Thunder returns ~500 posts
T+80ms:   Phoenix returns ~500 posts
T+85ms:   Hydrators start (parallel)
T+120ms:  All hydration complete
T+125ms:  Pre-scoring filters (sequential)
T+140ms:  Filters complete (~800 → ~600 candidates)
T+145ms:  Phoenix scorer calls ML model
T+180ms:  ML predictions returned
T+185ms:  Weighted scorer computes combined scores
T+190ms:  Author diversity scorer adjusts
T+195ms:  OON scorer adjusts
T+200ms:  Selector sorts and truncates to top 100
T+210ms:  Post-selection hydration (VF check)
T+230ms:  Post-selection filters
T+235ms:  Side effects spawned (background)
T+240ms:  Response returned to client
```

*Timings are approximate and vary by load.*

---

## External Dependencies

### Internal Services

| Service | Purpose | Protocol |
|---------|---------|----------|
| Thunder | In-network posts | gRPC |
| Phoenix (Retrieval) | Out-of-network candidates | gRPC |
| Phoenix (Ranking) | ML predictions | gRPC |
| Strato | User data storage | RPC |
| Gizmoduck | User profiles | gRPC |
| TweetEntityService | Media metadata | gRPC |
| VisibilityFiltering | Content moderation | gRPC |
| SocialGraph | Follow relationships | gRPC |

### Infrastructure

| Component | Purpose |
|-----------|---------|
| Kafka | Real-time post events |
| Strato | User data persistence |
| S3 | Model checkpoint storage |

### Python Dependencies (Phoenix)

```
jax / jaxlib          # ML framework
dm-haiku             # Neural network library
numpy                # Numerical computing
optax                # Optimizers (training)
```

### Rust Dependencies

```
tonic                # gRPC framework
tokio                # Async runtime
dashmap              # Concurrent hashmap
anyhow               # Error handling
log                  # Logging
```

---

## Configuration & Constants

### Pipeline Parameters (params.rs - excluded from OSS)

Key configuration values (names visible, values redacted):

- `MAX_POST_AGE`: Maximum age of posts to serve
- `RESULT_SIZE`: Number of posts to return
- `THUNDER_MAX_RESULTS`: Max posts from Thunder
- `PHOENIX_MAX_RESULTS`: Max candidates from Phoenix retrieval
- `MAX_GRPC_MESSAGE_SIZE`: gRPC message size limit

### Scoring Weights (params.rs - excluded from OSS)

Weights for combining action predictions:

- `FAVORITE_WEIGHT`: Weight for like prediction
- `REPLY_WEIGHT`: Weight for reply prediction
- `RETWEET_WEIGHT`: Weight for retweet prediction
- `QUOTE_WEIGHT`: Weight for quote prediction
- `CLICK_WEIGHT`: Weight for click prediction
- `PROFILE_CLICK_WEIGHT`: Weight for profile click
- `VQV_WEIGHT`: Weight for video quality view
- `PHOTO_EXPAND_WEIGHT`: Weight for photo expand
- `SHARE_WEIGHT`: Weight for share
- `DWELL_WEIGHT`: Weight for dwell
- `FOLLOW_AUTHOR_WEIGHT`: Weight for follow author
- `NOT_INTERESTED_WEIGHT`: Weight for not interested (negative)
- `BLOCK_AUTHOR_WEIGHT`: Weight for block (negative)
- `MUTE_AUTHOR_WEIGHT`: Weight for mute (negative)
- `REPORT_WEIGHT`: Weight for report (negative)

### Diversity Parameters

- `AUTHOR_DIVERSITY_DECAY`: Exponential decay for repeated authors
- `AUTHOR_DIVERSITY_FLOOR`: Minimum score multiplier

---

## Development Guidelines

### Adding a New Source

1. Implement `Source<ScoredPostsQuery, PostCandidate>` trait
2. Add `enable()` method for conditional execution
3. Return `Result<Vec<PostCandidate>, String>`
4. Register in `PhoenixCandidatePipeline::build_with_clients()`

### Adding a New Filter

1. Implement `Filter<ScoredPostsQuery, PostCandidate>` trait
2. Return `FilterResult { kept, removed }`
3. Keep order deterministic for reproducibility
4. Add appropriate logging/metrics

### Adding a New Scorer

1. Implement `Scorer<ScoredPostsQuery, PostCandidate>` trait
2. Must preserve candidate order and length
3. Store results in appropriate score field
4. Consider impact on downstream scorers

### Testing

- **Python**: Use `pytest` for ML components
- **Rust**: Add unit tests alongside implementation
- **Integration**: Test full pipeline with mock clients

### Code Style

- **Rust**: Standard `rustfmt` formatting
- **Python**: Follow PEP 8, use type hints
- **Comments**: Explain "why" not "what" for complex logic

---

## License

Apache License 2.0

---

## Summary of Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `candidate-pipeline/candidate_pipeline.rs` | Pipeline orchestration | ~330 |
| `home-mixer/candidate_pipeline/phoenix_candidate_pipeline.rs` | Concrete pipeline | ~256 |
| `home-mixer/server.rs` | gRPC service | ~84 |
| `home-mixer/scorers/phoenix_scorer.rs` | ML scoring | ~177 |
| `home-mixer/scorers/weighted_scorer.rs` | Score combination | ~93 |
| `thunder/posts/post_store.rs` | In-memory storage | ~527 |
| `thunder/thunder_service.rs` | gRPC API | ~340 |
| `phoenix/recsys_model.py` | Ranking model | ~475 |
| `phoenix/recsys_retrieval_model.py` | Retrieval model | ~373 |
| `phoenix/grok.py` | Transformer | ~587 |

---

## Contact & Contributing

This is an open-source release of the X For You Feed algorithm. For questions or issues, please refer to the repository's issue tracker.
