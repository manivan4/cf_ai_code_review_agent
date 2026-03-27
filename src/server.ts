import { createWorkersAI } from "workers-ai-provider";
import { routeAgentRequest, callable, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  streamText,
  convertToModelMessages,
  pruneMessages,
  tool,
  stepCountIs,
  type ModelMessage
} from "ai";
import { z } from "zod";

/**
 * The AI SDK's downloadAssets step runs `new URL(data)` on every file
 * part's string data. Data URIs parse as valid URLs, so it tries to
 * HTTP-fetch them and fails. Decode to Uint8Array so the SDK treats
 * them as inline data instead.
 */
function inlineDataUrls(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((msg) => {
    if (msg.role !== "user" || typeof msg.content === "string") return msg;
    return {
      ...msg,
      content: msg.content.map((part) => {
        if (part.type !== "file" || typeof part.data !== "string") return part;
        const match = part.data.match(/^data:([^;]+);base64,(.+)$/);
        if (!match) return part;
        const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
        return { ...part, data: bytes, mediaType: match[1] };
      })
    };
  });
}

export class ChatAgent extends AIChatAgent<Env, { preferences?: string }> {
  maxPersistedMessages = 100;
  initialState = { preferences: "" };

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    const result = streamText({
      model: workersai("@cf/meta/llama-4-scout-17b-16e-instruct", {
        sessionAffinity: this.sessionAffinity
      }),
      system: `You are a Developer Productivity Agent — a friendly, proactive AI assistant for software engineers. You always try to help, even with short or casual messages. Never say "your request is incomplete" — instead, do your best to interpret the user's intent and use your tools.

Your capabilities:
- Fetch real-time GitHub repository stats (stars, forks, issues)
- Check weather for any city
- Deep code analysis (security, complexity, anti-patterns)
- Summarize any website by URL
- Perform math calculations
- Schedule reminders and tasks
- Save and recall user preferences
- Deploy services to staging (requires user approval)
- Detect the user's timezone

When a user mentions a city and weather, use the getWeather tool. When they mention a GitHub repo, use getGitHubRepoStats. When they share a URL, use summarizeWebsite. When they share code, use analyzeCodeSnippet. When they ask you to remember something, use saveUserPreference. Be proactive and helpful!

User Preferences (Keep these in mind for your responses):
${this.state.preferences || "No preferences saved yet."}

${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the scheduleTask tool.`,
      // Prune old tool calls to save tokens on long conversations
      messages: pruneMessages({
        messages: inlineDataUrls(await convertToModelMessages(this.messages)),
        toolCalls: "before-last-2-messages"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        saveUserPreference: tool({
          description: "Save a user preference (e.g., tech stack, name, role) to your persistent memory.",
          inputSchema: z.object({
            preference: z.string().describe("The preference to remember")
          }),
          execute: async ({ preference }) => {
            const currentPrefs = this.state.preferences ? this.state.preferences + "\n" : "";
            const newPrefs = currentPrefs + "- " + preference;
            this.setState({ ...this.state, preferences: newPrefs });
            return `Saved preference: ${preference}`;
          }
        }),

        analyzeCodeSnippet: tool({
          description: "Perform deep static analysis on a code snippet to find bugs, anti-patterns, security issues, and estimate complexity.",
          inputSchema: z.object({
            code: z.string().describe("The code snippet to analyze")
          }),
          execute: async ({ code }) => {
            const lines = code.split("\n");
            const issues: string[] = [];
            const metrics: Record<string, number | string> = { linesOfCode: lines.length };

            // Complexity: count branches
            const branchKeywords = ["if", "else if", "case", "catch", "&&", "||", "?"];
            let complexity = 1;
            for (const kw of branchKeywords) complexity += (code.match(new RegExp(kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), "g")) || []).length;
            metrics.estimatedCyclomaticComplexity = complexity;
            if (complexity > 10) issues.push(`High complexity (${complexity}) — consider breaking into smaller functions`);

            // Security red flags
            if (code.includes("eval(")) issues.push("CRITICAL: eval() is a major security risk");
            if (code.includes("innerHTML")) issues.push("CRITICAL: innerHTML can lead to XSS vulnerabilities");
            if (/document\.write/.test(code)) issues.push("Warning: document.write is discouraged");

            // Anti-patterns
            if (code.includes("console.log")) issues.push("Cleanup: remove console.log before production");
            if (code.includes(": any")) issues.push("Type safety: avoid using 'any' in TypeScript");
            if (/\.then\(.*\.then\(/.test(code)) issues.push("Anti-pattern: nested .then() chains — use async/await instead");
            if (/callback.*callback|cb.*cb/i.test(code)) issues.push("Anti-pattern: possible callback hell detected");

            // Code smells
            const magicNumbers = code.match(/[^\w.]\d{2,}[^\w.]/g);
            if (magicNumbers && magicNumbers.length > 2) issues.push(`Code smell: ${magicNumbers.length} magic numbers found — use named constants`);
            if (/TODO|FIXME|HACK|XXX/i.test(code)) issues.push("Note: contains TODO/FIXME/HACK comments that need attention");
            if (lines.length > 30) issues.push("Readability: function exceeds 30 lines — consider refactoring");

            // Missing patterns
            if (code.includes("await ") && !code.includes("try")) issues.push("Robustness: async code without try/catch error handling");
            if (code.includes("fetch(") && !code.includes(".ok") && !code.includes("catch")) issues.push("Robustness: fetch() without response status or error checking");

            return {
              metrics,
              issues: issues.length > 0 ? issues : ["Excellent! No issues detected. Code looks clean and production-ready."]
            };
          }
        }),

        deployToStaging: tool({
          description: "Deploy the current codebase to the staging environment. Requires human approval.",
          inputSchema: z.object({
            serviceName: z.string().describe("Name of the service to deploy")
          }),
          needsApproval: async () => true,
          execute: async ({ serviceName }) => {
            return `Successfully deployed ${serviceName} to staging environment!`;
          }
        }),

        summarizeWebsite: tool({
          description: "Fetch a URL and return a summary of the page including title, description, and word count.",
          inputSchema: z.object({
            url: z.string().url().describe("The full URL to fetch and summarize")
          }),
          execute: async ({ url }) => {
            try {
              const res = await fetch(url, {
                headers: { "User-Agent": "Cloudflare-AI-Agent" }
              });
              if (!res.ok) return { error: `Failed to fetch: ${res.status} ${res.statusText}` };
              const html = await res.text();
              const titleMatch = html.match(/<title[^>]*>([^<]*)<\/title>/i);
              const descMatch = html.match(/<meta[^>]*name=["']description["'][^>]*content=["']([^"']*)["']/i);
              const textOnly = html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
              return {
                url,
                title: titleMatch ? titleMatch[1].trim() : "No title found",
                description: descMatch ? descMatch[1].trim() : "No meta description found",
                wordCount: textOnly.split(/\s+/).length,
                fetchedAt: new Date().toISOString()
              };
            } catch (error) {
              return { error: String(error) };
            }
          }
        }),

        getGitHubRepoStats: tool({
          description: "Fetch real-time statistics for a public GitHub repository.",
          inputSchema: z.object({
            owner: z.string().describe("The owner of the repository (e.g., 'cloudflare')"),
            repo: z.string().describe("The name of the repository (e.g., 'agents-starter')")
          }),
          execute: async ({ owner, repo }) => {
            try {
              const res = await fetch(`https://api.github.com/repos/${owner}/${repo}`, {
                headers: { "User-Agent": "Cloudflare-AI-Agent" }
              });
              if (!res.ok) {
                return { error: `GitHub API returned ${res.status}: ${res.statusText}` };
              }
              const data: any = await res.json();
              return {
                name: data.full_name,
                description: data.description,
                stars: data.stargazers_count,
                forks: data.forks_count,
                open_issues: data.open_issues_count,
                language: data.language
              };
            } catch (error) {
              return { error: String(error) };
            }
          }
        }),

        // Server-side tool: runs automatically on the server
        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            // Replace with a real weather API in production
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius"
            };
          }
        }),

        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
