import { Link } from "react-router-dom";

interface LogoProps {
  className?: string;
  variant?: "full" | "symbol" | "stacked";
  showSubtitle?: boolean;
  linkTo?: string;
}

export function Logo({
  className = "",
  variant = "full",
  showSubtitle = false,
  linkTo = "/",
}: LogoProps) {
  const content = (
    <div className={`flex items-center gap-2.5 select-none ${className}`}>
      {variant === "symbol" ? (
        <>
          <img
            src="/swyra-symbol.png"
            alt="SWYRA"
            className="h-7 w-auto object-contain dark:hidden"
          />
          <img
            src="/swyra-symbol-light.png"
            alt="SWYRA"
            className="h-7 w-auto object-contain hidden dark:block"
          />
        </>
      ) : variant === "stacked" ? (
        <div className="flex flex-col items-center gap-1.5">
          <img
            src="/swyra-symbol.png"
            alt="SWYRA"
            className="h-10 w-auto object-contain dark:hidden"
          />
          <img
            src="/swyra-symbol-light.png"
            alt="SWYRA"
            className="h-10 w-auto object-contain hidden dark:block"
          />
          <img
            src="/swyra-wordmark.png"
            alt="SWYRA"
            className="h-4 w-auto object-contain dark:hidden"
          />
          <img
            src="/swyra-wordmark-light.png"
            alt="SWYRA"
            className="h-4 w-auto object-contain hidden dark:block"
          />
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <img
            src="/swyra-symbol.png"
            alt="SWYRA"
            className="h-6 w-auto object-contain dark:hidden"
          />
          <img
            src="/swyra-symbol-light.png"
            alt="SWYRA"
            className="h-6 w-auto object-contain hidden dark:block"
          />
          <img
            src="/swyra-wordmark.png"
            alt="SWYRA"
            className="h-4 w-auto object-contain dark:hidden"
          />
          <img
            src="/swyra-wordmark-light.png"
            alt="SWYRA"
            className="h-4 w-auto object-contain hidden dark:block"
          />
          {showSubtitle && (
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground border-l border-border pl-2 ml-1">
              Analytics
            </span>
          )}
        </div>
      )}
    </div>
  );

  if (linkTo) {
    return (
      <Link to={linkTo} className="inline-flex items-center focus:outline-none hover:opacity-90 transition-opacity">
        {content}
      </Link>
    );
  }

  return content;
}

export default Logo;
