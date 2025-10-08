# Common Mistakes Log

This file tracks mistakes made during development to avoid repeating them.

## Mistakes Made

### [Date: 2025-10-08]

#### Mistake 1: [Pending - Will be updated as work progresses]
- **What happened**: TBD
- **Why it happened**: TBD
- **How to avoid**: TBD

---

## Best Practices Reminder

1. **Always read files before editing** - Use Read tool first
2. **Test configuration** - Verify settings before running analyzers
3. **Use pathlib.Path** - For Windows compatibility
4. **Async operations** - Use asyncio.to_thread for CPU-bound tasks
5. **Error handling** - Return success: false with details, don't crash
6. **Type hints** - Always include for clarity
7. **Dataclass serialization** - Provide to_dict() methods
8. **Protected content** - Always check before flagging grammar issues
9. **Batch LLM calls** - Reduce API calls where possible
10. **Clean, structured, simple** - Follow Amida code standards

---

*This file will be updated throughout development*
