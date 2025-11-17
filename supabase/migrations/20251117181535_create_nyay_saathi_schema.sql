/*
  # Nyay-Saathi Database Schema

  1. New Tables
    - `users` - Track user sessions and preferences
    - `conversations` - Store chat conversations with metadata
    - `messages` - Individual chat messages with sources
    - `documents` - User-uploaded legal documents
    - `feedback` - User feedback on responses
    - `analytics` - Track API usage and performance

  2. Security
    - RLS enabled on all tables
    - Policies allow read/write access to own data
    - Service role access for analytics

  3. Features
    - Automatic timestamps for all records
    - Soft delete support (deleted_at column)
    - Message source tracking (guides vs documents)
    - Response caching for identical questions
*/

-- Users table
CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id text UNIQUE NOT NULL,
  language text DEFAULT 'Simple English',
  created_at timestamptz DEFAULT now(),
  last_active timestamptz DEFAULT now(),
  deleted_at timestamptz
);

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own data"
  ON users FOR SELECT
  TO anon
  USING (session_id = current_setting('app.session_id', true));

CREATE POLICY "Users can update own data"
  ON users FOR UPDATE
  TO anon
  USING (session_id = current_setting('app.session_id', true))
  WITH CHECK (session_id = current_setting('app.session_id', true));

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title text DEFAULT 'Untitled Conversation',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  deleted_at timestamptz
);

ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own conversations"
  ON conversations FOR SELECT
  TO anon
  USING (user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true)));

CREATE POLICY "Users can create conversations"
  ON conversations FOR INSERT
  TO anon
  WITH CHECK (user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true)));

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('user', 'assistant')),
  content text NOT NULL,
  sources_from_guides text[] DEFAULT '{}',
  source_from_document boolean DEFAULT false,
  feedback text CHECK (feedback IN ('positive', 'negative', null)),
  created_at timestamptz DEFAULT now(),
  deleted_at timestamptz
);

ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own messages"
  ON messages FOR SELECT
  TO anon
  USING (conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true))));

CREATE POLICY "Users can create messages"
  ON messages FOR INSERT
  TO anon
  WITH CHECK (conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true))));

CREATE POLICY "Users can update own messages"
  ON messages FOR UPDATE
  TO anon
  USING (conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true))))
  WITH CHECK (conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true))));

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  conversation_id uuid REFERENCES conversations(id) ON DELETE CASCADE,
  file_name text NOT NULL,
  file_type text NOT NULL,
  file_size integer,
  raw_text text NOT NULL,
  explanation text,
  language text DEFAULT 'Simple English',
  created_at timestamptz DEFAULT now(),
  deleted_at timestamptz
);

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own documents"
  ON documents FOR SELECT
  TO anon
  USING (user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true)));

CREATE POLICY "Users can create documents"
  ON documents FOR INSERT
  TO anon
  WITH CHECK (user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true)));

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id uuid NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  rating text NOT NULL CHECK (rating IN ('positive', 'negative')),
  comment text,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own feedback"
  ON feedback FOR SELECT
  TO anon
  USING (user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true)));

CREATE POLICY "Users can create feedback"
  ON feedback FOR INSERT
  TO anon
  WITH CHECK (user_id IN (SELECT id FROM users WHERE session_id = current_setting('app.session_id', true)));

-- Analytics table (service role only)
CREATE TABLE IF NOT EXISTS analytics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES users(id) ON DELETE SET NULL,
  event_type text NOT NULL,
  event_data jsonb DEFAULT '{}',
  response_time_ms integer,
  api_used text,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE analytics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role only"
  ON analytics FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_session_id ON users(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at);
