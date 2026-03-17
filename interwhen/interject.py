import asyncio
import httpx
import json
import logging

logger = logging.getLogger(__name__)


async def _cancel_tasks(tasks):
    """Cancel a list of asyncio Tasks and await them (swallow CancelledError)."""
    if not tasks:
        return
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

async def stream_completion(prompt, prev_text = "", llm_server=None, monitors=[], add_delay=False, 
            num_calls_index=0, termination_requires_validation=False, async_execution=True):
    stop_event = asyncio.Event()
    stop_info = {"generated_text": None, "feedback": None, "token_index": None}
    monitor_tasks = []

    logger.warning("="*50 + f"Calling LM with prompt (call #{num_calls_index})" + "="*50)
    generated_text = prev_text
    llm_server["payload"]["prompt"] = prompt + prev_text

    logger.info(f"#{num_calls_index}: EOS")
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", llm_server["url"], headers=llm_server["headers"], json=llm_server["payload"]) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    else:
                        # Obtain the current token (text chunk)
                        try:
                            chunk = json.loads(data)["choices"][0]["text"]
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            logger.debug(f"Skipping malformed SSE data: {data!r} ({e})")
                            continue
                        # If any event is already set, break immediately (we don't want more chunks)
                        if stop_event.is_set():
                            logger.info(f'\n[Early stop already triggered, ignoring chunk: {chunk}]')
                            break
                        print(chunk, end="", flush=True)
                        generated_text += chunk

                        # Start monitor task in background with chunk index
                        if len(monitors) > 0 and not stop_event.is_set():
                            stepFlag, step = monitors[0].step_extractor(chunk, generated_text)
                            if stepFlag:
                                if not (stop_event.is_set()):
                                    task = asyncio.create_task(monitors[0].verify(step, len(generated_text) - len(chunk), stop_event, stop_info))
                                    monitor_tasks.append(task)
                                    if not async_execution:
                                        await task
                        if add_delay:
                            await asyncio.sleep(0.1)

    # If any monitor event fired, cancel remaining monitor tasks right away (don't wait for them to finish).
    if len(monitors) > 0 and async_execution:
        if stop_event.is_set():
            logger.debug("Monitor event detected — cancelling pending monitor tasks immediately.")
            await _cancel_tasks(monitor_tasks)
        else:
            await asyncio.gather(*monitor_tasks, return_exceptions=True)

    if stop_event.is_set(): 
        if num_calls_index >= 50:
            logger.info("\n\\n[Maximum correction attempts reached. Stopping generation.]")
            return generated_text

        corrected_text = await monitors[0].fix(generated_text, stop_info)
        if stop_info["feedback"] == "\nthe answer is \\boxed{no solution}":
            return corrected_text # No solution found, return no solution ie soundness is 100% is it doesnt pass the verifer
        if stop_info.get("phase") == "final_answer_correct":
            return corrected_text  # Expression verified correct, stop generation
        return await stream_completion(prompt, prev_text=corrected_text, llm_server=llm_server, monitors=monitors, add_delay=add_delay, num_calls_index=num_calls_index+1, termination_requires_validation=termination_requires_validation, async_execution=async_execution)

    return generated_text