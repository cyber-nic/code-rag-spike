// Package main provides a simple example of how to use the hnsw.
package main

import (
	"bufio"
	"context"
	"crypto/md5"
	"database/sql"
	"encoding/hex"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	embed "github.com/codectx/tokens/services/embed"
	store "github.com/codectx/tokens/services/store"
	goignore "github.com/cyber-nic/go-gitignore"
	ollama "github.com/ollama/ollama/api"

	"github.com/coder/hnsw"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// ContextKey is an alias for string type
type ContextKey string

const (
	// LoggerCtxKey is the string used to extract logger
	LoggerCtxKey ContextKey = "logger"
)

func main() {
	begin := time.Now()

	mu := sync.Mutex{}

	wd, query, err := getWorkingDirAndQuery(os.Args)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	slog.Debug("begin", "path", wd, "query", query)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := &slog.HandlerOptions{
		AddSource: true,
		// Level:     slog.LevelDebug,
		ReplaceAttr: func(_ []string, a slog.Attr) slog.Attr {
			if a.Key == slog.TimeKey {
				a.Key = "ts"
				a.Value = slog.Int64Value(time.Now().Unix())
			}
			if a.Key == slog.SourceKey {
				source, _ := a.Value.Any().(*slog.Source)
				if source != nil {
					source.File = filepath.Base(source.File)
				}
			}
			return a
		},
	}
	handler := slog.NewTextHandler(os.Stdout, opts)
	l := slog.New(handler)

	ctx = context.WithValue(ctx, LoggerCtxKey, l)

	// read voyageAPIKeyPath from env var
	// voyageAPIKeyPath := os.Getenv("VOYAGE_API_KEY_FILE")
	// if voyageAPIKeyPath == "" {
	// 	l.Error("VOYAGE_API_KEY_FILE env var not set")
	// 	os.Exit(1)
	// }

	// Load API key from file
	// voyageKey, err := os.ReadFile(voyageAPIKeyPath)
	// if err != nil {
	// 	l.Error("Failed to read API key", "error", err)
	// }
	// vKey := strings.TrimSpace(string(voyageKey))

	// Setup logger
	globIgnorePatterns, err := goignore.CompileIgnoreFile(".astignore")

	database, err := sql.Open("duckdb", "local.db")
	if err != nil {
		l.Error("Failed to connect to DuckDB", "error", err)
		os.Exit(1)
	}
	defer database.Close()

	// Setup storage service
	db := store.NewStorageService(database)

	// Setup Ollama
	os.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	oClient, err := ollama.ClientFromEnvironment()
	if err != nil {
		l.Error("Failed to create Ollama client", "error", err)
		os.Exit(1)
	}

	// Setup tokenizer to measure tokens
	configFile, err := tokenizer.CachedPath("bert-base-uncased", "tokenizer.json")
	if err != nil {
		l.Error("Failed to get cached path", "error", err)
		os.Exit(1)
	}
	tk, err := pretrained.FromFile(configFile)
	if err != nil {
		l.Error("Failed to load tokenizer", "error", err)
		os.Exit(1)
	}

	// Create embedding service
	emb := embed.NewEmbedService(oClient, tk)

	// Search
	q, _, err := emb.Get(ctx, query)
	// q, _, err := emb.Voyage(vKey, query)
	if err != nil {
		l.Error("Failed to embed query", "error", err)
		return
	}

	// make string channel of 5
	indexing := make(chan string, 5)

	done := atomic.Bool{}
	done.Store(false)

	numWorkers := 4

	g := hnsw.NewGraph[string]()

	// create wait group for workers
	var wg sync.WaitGroup

	// create 4 go routine workers that read from the indexing channel to perform work
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(id int) {
			defer l.Debug("worker", "id", id, "state", "done")
			defer wg.Done()

			for {
				select {
				case path := <-indexing:
					if err := handleFile(ctx, &mu, db, emb, g, q, path); err != nil {
						l.Error("Failed to handle file", "error", err)
					}
				}

				if done.Load() {
					return
				}

			}
		}(i)
	}

	// Walk through all files in the current directory
	filepath.Walk(wd, func(path string, info os.FileInfo, err error) error {
		// Skip directories
		info, err = os.Stat(path)
		if err != nil || info.IsDir() {
			return nil
		}
		// Skip files that match the ignore patterns
		if globIgnorePatterns.MatchesPath(path) {
			return nil
		}

		indexing <- path

		return nil
	})

	// Inform workers that there is no more work
	done.Store(true)
	l.Debug("done walking the tree")

	// Wait for all workers to finish
	wg.Wait()

	// Display
	neighbors := g.Search(q, 1)
	for _, n := range neighbors {
		d1, d2, d3 := getDistance(q, n.Value)
		l.Info("neighbour", "path", n.Key, "d1", d1, "d2", d2, "d3", d3)
	}

	fmt.Println(time.Since(begin).Milliseconds())
}

func getDistance(q, v []float32) (float32, float32, float32) {
	var sum float32
	for i := range q {
		sum += (q[i] - v[i]) * (q[i] - v[i])
	}

	d1 := hnsw.CosineDistance(q, v)
	d2 := hnsw.EuclideanDistance(q, v)
	return d1, d2, sum
}

// computeHash returns the MD5 hash of the given data
func computeHash(data []byte) string {
	hasher := md5.New()
	hasher.Write(data)
	return hex.EncodeToString(hasher.Sum(nil))
}

// handleFile reads the file at the given path, computes its hash, and embeds its content.
func handleFile(ctx context.Context, mu *sync.Mutex, db store.StorageService, emb embed.EmbeddingService, g *hnsw.Graph[string], q []float32, path string) error {
	l := ctx.Value(LoggerCtxKey).(*slog.Logger)
	start := time.Now()

	// read file content
	f, err := os.ReadFile(path)
	if err != nil {
		l.Error("Failed to read file", "error", err)
		os.Exit(1)
	}

	// Compute content hash
	hash := computeHash(f)

	// Determine if file has changed
	match, err := db.MatchHash(ctx, path, hash)
	if err != nil {
		l.Error("Failed to compare hash", "error", err)
		return nil
	}

	// If hash is the same, file has not changed
	if match {
		// get from db
		b, err := db.Get(ctx, []string{path})
		if err != nil {
			l.Error("Failed to get embedding", "error", err)
			return nil
		}
		// Add to graph
		mu.Lock()
		g.Add(hnsw.MakeNode(path, b[0].Vector))
		mu.Unlock()

		// Skip
		d1, d2, _ := getDistance(q, b[0].Vector)
		l.Debug("match", "path", path, "d1", d1, "d2", d2)
		return nil
	}

	// Embed
	vec, meta, err := emb.Get(ctx, string(f))
	// vec, meta, err := emb.Voyage(vKey, string(f))
	if err != nil {
		l.Error("Failed to embed text", "error", err)
		return nil
	}

	// Upsert
	if err := db.Upsert(ctx, path, hash, vec); err != nil {
		l.Error("Failed to create embedding", "error", err)
		return nil
	}

	// Add to graph
	mu.Lock()
	g.Add(hnsw.MakeNode(path, vec))
	mu.Unlock()

	d1, d2, _ := getDistance(q, vec)

	l.Debug("diff", "path", path, "d1", d1, "d2", d2, "emb_ms", meta.Duration, "tokens", meta.Tokens, "total_ms", time.Since(start).Milliseconds())
	return nil
}

// promptForUserQuery prompts the user to input a search query
func promptForUserQuery() (string, error) {
	fmt.Printf("Query: ")
	q, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil {
		return "", err
	}

	return q, nil
}

// getWorkingDirAndQuery returns the working directory and query from the command line arguments.
func getWorkingDirAndQuery(args []string) (string, string, error) {

	const usage = "Usage: ./main <optional:path> <optional:query>"

	if len(args) < 1 || len(args) > 3 {
		return "", "", fmt.Errorf(usage)
	}

	workingDir := "."
	query := ""

	if len(args) == 3 {
		workingDir = args[1]
		query = args[2]
	} else if len(args) == 2 {
		workingDir = args[1]

		var err error
		query, err = promptForUserQuery()
		if err != nil {
			return "", "", err
		}
	} else if len(args) == 1 {
		var err error
		query, err = promptForUserQuery()
		if err != nil {
			return "", "", err
		}
	}

	if query == "" {
		return "", "", fmt.Errorf(usage)
	}

	// test working directory
	if _, err := os.Stat(workingDir); err != nil {
		fmt.Printf("Invalid working directory: %s\n", workingDir)
		os.Exit(1)
	}

	return workingDir, query, nil
}

type stackFrame struct {
	Func   string `json:"func"`
	Source string `json:"source"`
	Line   int    `json:"line"`
}
