import React, { useMemo } from "react";

const HEADING_RE = /^#{1,6}\s+(.*)$/;
const LIST_RE = /^[-*]\s+(.*)$/;
const BOLD_SEGMENT_RE = /(\*\*[^*]+\*\*)/g;

function stripInlineMarkdown(text) {
  return text.replace(/\*\*/g, "").trim();
}

function renderInline(text, keyBase) {
  const parts = text.split(BOLD_SEGMENT_RE).filter(Boolean);
  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={`${keyBase}-${idx}`} className="font-semibold text-slate-900">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <React.Fragment key={`${keyBase}-${idx}`}>{stripInlineMarkdown(part)}</React.Fragment>;
  });
}

function parseNarrativeBlocks(text) {
  const lines = (text || "").replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let listItems = [];

  const flushList = () => {
    if (listItems.length > 0) {
      blocks.push({ type: "list", items: listItems });
      listItems = [];
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (!line) {
      flushList();
      if (blocks.length > 0 && blocks[blocks.length - 1].type !== "spacer") {
        blocks.push({ type: "spacer" });
      }
      continue;
    }

    const headingMatch = line.match(HEADING_RE);
    if (headingMatch) {
      flushList();
      blocks.push({ type: "heading", text: headingMatch[1].trim() });
      continue;
    }

    const listMatch = line.match(LIST_RE);
    if (listMatch) {
      listItems.push(listMatch[1].trim());
      continue;
    }

    flushList();
    blocks.push({ type: "paragraph", text: line });
  }

  flushList();

  while (blocks.length > 0 && blocks[0].type === "spacer") {
    blocks.shift();
  }
  while (blocks.length > 0 && blocks[blocks.length - 1].type === "spacer") {
    blocks.pop();
  }

  return blocks;
}

function blocksToPlainText(blocks) {
  const lines = [];
  for (const block of blocks) {
    if (block.type === "heading") {
      lines.push(stripInlineMarkdown(block.text));
      continue;
    }

    if (block.type === "paragraph") {
      lines.push(stripInlineMarkdown(block.text));
      continue;
    }

    if (block.type === "list") {
      for (const item of block.items) {
        lines.push(`- ${stripInlineMarkdown(item)}`);
      }
      continue;
    }

    lines.push("");
  }

  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

export default function NarrativePanel({ narrative, loading, onRegenerate, onCopy }) {
  const blocks = useMemo(() => parseNarrativeBlocks(narrative || ""), [narrative]);
  const cleanedText = useMemo(() => blocksToPlainText(blocks), [blocks]);

  return (
    <div className="card space-y-3">
      <div className="flex justify-between items-center">
        <h3 className="font-semibold">Panel C - Policy Narrative</h3>
      </div>

      {loading ? (
        <div className="animate-pulse text-muted">Generating policy narrative...</div>
      ) : cleanedText ? (
        <div className="text-sm text-slate-700">
          {blocks.map((block, idx) => {
            if (block.type === "heading") {
              return (
                <h4 key={`heading-${idx}`} className="font-semibold text-slate-900 mt-2 mb-1">
                  {renderInline(block.text, `heading-${idx}`)}
                </h4>
              );
            }

            if (block.type === "paragraph") {
              return (
                <p key={`paragraph-${idx}`} className="leading-6 mb-2">
                  {renderInline(block.text, `paragraph-${idx}`)}
                </p>
              );
            }

            if (block.type === "list") {
              return (
                <ul key={`list-${idx}`} className="list-disc pl-5 space-y-1 leading-6 mb-2">
                  {block.items.map((item, itemIdx) => (
                    <li key={`item-${idx}-${itemIdx}`}>{renderInline(item, `item-${idx}-${itemIdx}`)}</li>
                  ))}
                </ul>
              );
            }

            return <div key={`spacer-${idx}`} className="h-1" />;
          })}
        </div>
      ) : (
        <div className="text-sm text-muted">No narrative yet.</div>
      )}

      <div className="flex gap-2">
        <button className="px-3 py-2 bg-accent text-white rounded" onClick={onRegenerate}>
          Regenerate Narrative
        </button>
        <button className="px-3 py-2 border rounded" onClick={() => onCopy?.(cleanedText)}>
          Copy
        </button>
      </div>
    </div>
  );
}
