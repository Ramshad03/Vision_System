import subprocess
import speech_recognition as sr
import time
import os

# ─── CONFIGURATION ───────────────────────────────────────────────
TTS_RATE         = 165
STT_TIMEOUT      = 8
STT_PHRASE_LIMIT = 6
MAX_RETRIES      = 3
TTS_COOLDOWN     = 0.6
STT_LANGUAGE     = os.getenv("STT_LANGUAGE", "en-IN")
VOICE_MIC_INDEX  = os.getenv("VOICE_MIC_INDEX")


class VoiceInteraction:

    def __init__(self):
        self.recogniser = sr.Recognizer()
        self.recogniser.energy_threshold         = 200
        self.recogniser.pause_threshold          = 1.0
        self.recogniser.dynamic_energy_threshold = True
        self.recogniser.non_speaking_duration    = 0.5

        self.device_index = None
        if VOICE_MIC_INDEX not in (None, ""):
            try:
                self.device_index = int(VOICE_MIC_INDEX)
            except ValueError:
                print(f"[Voice] Invalid VOICE_MIC_INDEX '{VOICE_MIC_INDEX}', using default microphone.")

        mics = sr.Microphone.list_microphone_names()
        print("[Voice] Module initialised.")
        print(f"[Voice] STT language: {STT_LANGUAGE}")
        print(f"[Voice] Microphone index: {self.device_index if self.device_index is not None else 'system default'}")
        print("[Voice] Available microphones:")
        for i, name in enumerate(mics):
            print(f"         [{i}] {name}")

    # ─── SPEAK ───────────────────────────────────────────────────
    def speak(self, text):
        """
        Uses Windows PowerShell SAPI to speak text.
        Fully blocks until speech completes.
        """
        print(f"[Voice] Speaking: \"{text}\"")

        safe_text  = text.replace("'", "''")
        ps_command = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = 1; "
            f"$s.Volume = 100; "
            f"$s.Speak('{safe_text}'); "
            f"$s.Dispose()"
        )

        try:
            subprocess.run(
                ["powershell", "-Command", ps_command],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            print(f"[Voice] TTS error: {e}")
        except FileNotFoundError:
            print("[Voice] ERROR: PowerShell not found.")

        time.sleep(TTS_COOLDOWN)

    # ─── LISTEN ──────────────────────────────────────────────────
    def listen(self, device_index=None):
        """
        Listens via microphone and returns transcribed text.
        Retries up to MAX_RETRIES times.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[Voice] Listening... (attempt {attempt}/{MAX_RETRIES})")

                selected_device = self.device_index if device_index is None else device_index
                with sr.Microphone(device_index=selected_device) as source:
                    if attempt == 1:
                        print("[Voice] Calibrating for ambient noise...")
                        self.recogniser.adjust_for_ambient_noise(source, duration=0.8)
                        print(f"[Voice] Energy threshold: {self.recogniser.energy_threshold:.0f}")

                    print("[Voice] Speak now...")
                    audio = self.recogniser.listen(
                        source,
                        timeout=STT_TIMEOUT,
                        phrase_time_limit=STT_PHRASE_LIMIT
                    )

                print("[Voice] Processing speech...")
                text = self.recogniser.recognize_google(audio, language=STT_LANGUAGE)
                text = text.strip().lower()
                print(f"[Voice] Heard: \"{text}\"")
                return text

            except sr.WaitTimeoutError:
                print(f"[Voice] No speech detected (attempt {attempt}).")
                if attempt < MAX_RETRIES:
                    self.speak("I did not hear anything. Please try again.")

            except sr.UnknownValueError:
                print(f"[Voice] Could not understand audio (attempt {attempt}).")
                if attempt < MAX_RETRIES:
                    self.speak("Sorry, I did not catch that. Please try again.")

            except sr.RequestError as e:
                print(f"[Voice] Google STT error: {e}")
                break

        print("[Voice] All retries exhausted.")
        return None

    def _typed_fallback(self, prompt):
        """
        Lets registration continue when microphone/STT is unreliable.
        Runs in the terminal where the app was started.
        """
        try:
            typed = input(f"[Voice] {prompt} (type here, or press Enter to cancel): ").strip()
        except EOFError:
            return None

        return typed.lower() if typed else None

    def _clean_name(self, raw):
        text = raw.strip()
        lower = text.lower()
        prefixes = (
            "my name is ",
            "i am ",
            "i'm ",
            "this is ",
            "name is ",
            "call me ",
        )

        for prefix in prefixes:
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        return text.title() if text else None

    def _confirm(self, name):

        # ─── POSITIVE CONFIRMATION WORDS ─────────────────────────
        POSITIVE = {
            "yes", "yeah", "yep", "yup", "sure", "correct", "right",
            "positive", "exactly", "affirmative", "true", "ok", "okay",
            "perfect", "good", "fine", "confirm", "confirmed", "that's right",
            "thats right", "absolutely", "indeed", "certainly", "of course"
        }

        # ─── NEGATIVE CONFIRMATION WORDS ─────────────────────────
        NEGATIVE = {
            "no", "nope", "nah", "wrong", "incorrect", "negative",
            "false", "not right", "thats wrong", "that's wrong",
            "change", "different", "not", "never", "nay"
        }

        self.speak(f"I heard {name}. Is that correct?")
        response = self.listen()

        if response is None:
            response = self._typed_fallback(f"Confirm '{name}'? Type yes or no")
            if response is None:
                self.speak("I did not catch that. I will try again next time.")
                return False

        
        response_words = set(response.lower().replace("'", "").replace(",", "").split())

        if response_words & POSITIVE:
            
            return True

        elif response_words & NEGATIVE:
            
            self.speak("Sorry about that. Please try again.")
            return False

        else:
            # ─── UNCLEAR RESPONSE ─────────────────────────────────
            self.speak("I could not understand your answer. I will try again next time.")
            return False

    # ─── HUMAN REGISTRATION DIALOGUE ─────────────────────────────
    def ask_for_name(self):
        """
        Full voice dialogue to register an unknown person.
        Asks for name → confirms → saves only if confirmed.

        Returns:
            name : str — Title Case confirmed name, or None
        """
        self.speak("Hello! I do not recognise you. What is your name?")

        raw = self.listen()
        if raw is None:
            raw = self._typed_fallback("Could not hear your name")
            if raw is None:
                self.speak("Sorry, I could not catch your name. I will try again next time.")
                return None

        name = self._clean_name(raw)
        if name is None:
            return None

        # ─── CONFIRM BEFORE SAVING ────────────────────────────────
        confirmed = self._confirm(name)
        if not confirmed:
            return None

        self.speak(f"Nice to meet you, {name}. I will remember you from now on.")
        return name

    # ─── ANIMAL REGISTRATION DIALOGUE ────────────────────────────
    def ask_for_animal_name(self, yolo_label):
        """
        Full voice dialogue to name an unknown animal.
        Asks for name → confirms → saves only if confirmed.

        Args:
            yolo_label : str — YOLO class (e.g. 'cat', 'dog')

        Returns:
            name : str — Title Case confirmed animal name, or None
        """
        self.speak(
            f"I have detected a {yolo_label}. "
            f"What is the name of your {yolo_label}?"
        )

        raw = self.listen()
        if raw is None:
            raw = self._typed_fallback(f"Could not hear the {yolo_label}'s name")
            if raw is None:
                self.speak("Sorry, I could not catch the name. I will try again next time.")
                return None

        name = self._clean_name(raw)
        if name is None:
            return None

        # ─── CONFIRM BEFORE SAVING ────────────────────────────────
        confirmed = self._confirm(name)
        if not confirmed:
            return None

        self.speak(f"Got it! I will remember this {yolo_label} as {name}.")
        return name
